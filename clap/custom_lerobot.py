import yaml
import time
import logging
from pathlib import Path
from typing import List, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import cv2
import torch
import datasets
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import IterableDataset, Dataset
# lerobot is pinned to the legacy v0.1.0 fork in requirements.txt — see the
# `lerobot @ git+...` line for the exact commit hash. That snapshot still has
# the `lerobot.common.datasets` layout that this dataset wrapper was written
# against; do NOT pip install a newer lerobot release without porting these
# imports over to the new `lerobot.datasets` package layout.
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from clap.data_transform import AstribotPipeline
from clap.data_transform_droid import (
    DROID_EXTERNAL_IMAGE_KEYS,
    DROID_WRIST_IMAGE_KEY,
    DroidPipeline,
)

ROBOT_ID_MAPPING = {
    "S1-stationary": 0,
    "agibot-go1": 1,
    "human": 2,
    "franka": 3,  # Libero
    "google_robot": 4,  # fractal
    "widowx": 5,  # bridge
    "droid": 6,
}

ARM_LAYOUT_BY_ROBOT_TYPE = {
    "S1-stationary": "dual",
    "agibot-go1": "dual",
    "human": "dual",
    "franka": "single_right",
    "google_robot": "single_right",
    "widowx": "single_right",
    "droid": "single_right",
}


def is_droid_repo(repo_id: str | Path) -> bool:
    return Path(repo_id).name == "droid_1.0.1_lerobot"


def robot_type_for_dataset(repo_id: str | Path, meta_robot_type: str) -> str:
    if is_droid_repo(repo_id):
        return "droid"
    return meta_robot_type


def _repo_info(repo_id: str | Path) -> dict:
    info_path = Path(repo_id) / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _repo_feature_keys(repo_id: str | Path) -> set[str]:
    info = _repo_info(repo_id)
    return set(info["features"].keys())


def _filter_delta_timestamps_for_repo(delta_timestamps: dict | None, repo_id: str | Path) -> dict | None:
    if delta_timestamps is None:
        return None

    info = _repo_info(repo_id)
    feature_keys = set(info["features"].keys())
    fps = float(info.get("fps", 30))
    filtered = {}
    for key, value in delta_timestamps.items():
        if key not in feature_keys:
            continue

        # Config values are frame-step indices. Convert them to seconds using
        # the fps of the target repo so chunk_size=32 always samples 32 steps.
        values = list(value)
        if all(float(v).is_integer() for v in values):
            filtered[key] = [float(v) / fps for v in values]
        else:
            filtered[key] = [float(v) for v in values]
    return filtered or None


def _droid_episode_file(root: Path, episode_index: int, key: str | None = None, suffix: str = "parquet") -> Path:
    chunk = f"chunk-{episode_index // 1000:03d}"
    filename = f"episode_{episode_index:06d}.{suffix}"
    if key is None:
        return root / "data" / chunk / filename
    return root / "videos" / chunk / key / filename


def get_available_droid_episodes(repo_id: str | Path, delta_timestamps: dict | None) -> list[int]:
    root = Path(repo_id)
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return []

    requested_video_keys = [
        key for key in (delta_timestamps or {})
        if key.startswith("observation.images.")
    ]
    valid_episodes = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            episode = yaml.safe_load(line)
            tasks = episode.get("tasks") or []
            if not any(str(task).strip() for task in tasks):
                continue

            episode_index = int(episode["episode_index"])
            if not _droid_episode_file(root, episode_index).is_file():
                continue

            missing_video = False
            for key in requested_video_keys:
                if not _droid_episode_file(root, episode_index, key, suffix="mp4").is_file():
                    missing_video = True
                    break
            if missing_video:
                continue

            valid_episodes.append(episode_index)
    return valid_episodes

def decode_video_frames_opencv(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float = 0.07,
    fps: int = 30,
    **kwargs
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    while isinstance(timestamps, list) and len(timestamps) == 1 and isinstance(timestamps[0], list):
        timestamps = timestamps[0]

    frames = []
    for ts in timestamps:
        frame_id = int(round(ts * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(tensor_frame)

    cap.release()
    return torch.stack(frames)


class CustomLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, local_files_only: bool | None = None, **kwargs) -> None:
        # ``local_files_only`` was a v2.0 kwarg that lerobot dropped in v2.1.
        # We accept and silently ignore it for backward compat with the
        # original UniVLA-era CLAP code; passing ``root`` already implies
        # offline use of an on-disk dataset.
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_ids = item["task_index"]
        if task_ids.ndim == 1:
            task_ids = task_ids.numpy().tolist()
            item["task"] = self.meta.tasks[task_ids[0]]
            item["subtask"] = self.meta.tasks[task_ids[1]]
        else:
            task_ids = task_ids.item()
            item["task"] = f"[subtask] {self.meta.tasks[task_ids]}"
            item["subtask"] = self.meta.tasks[task_ids]
        
        # Add metadata
        item["dataset_name"] = self.repo_id.split("/")[-1]
        robot_type = robot_type_for_dataset(self.root, self.meta.robot_type)
        item["robot_type"] = robot_type
        item["robot_id"] = ROBOT_ID_MAPPING[robot_type]
        item["arm_layout"] = ARM_LAYOUT_BY_ROBOT_TYPE.get(robot_type, "dual")

        return item

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            # else:
            #     query_timestamps[key] = [current_ts]
            # NOTE: we only query the keys from query_indices to save time

        return query_timestamps
    
    # def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
    #     """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
    #     in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
    #     Segmentation Fault. This probably happens because a memory reference to the video loader is created in
    #     the main process and a subprocess fails to access it.
    #     """
    #     item = {}
    #     for vid_key, query_ts in query_timestamps.items():
    #         video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
    #         frames = decode_video_frames_opencv(video_path, query_ts, self.tolerance_s, self.fps)
    #         item[vid_key] = frames.squeeze(0)

    #     return item


class MultiCustomLeRobotDataset(IterableDataset):
    """An IterableDataset consisting of multiple underlying `LeRobotDataset`s with efficient two-stage sampling.

    This class implements streaming data loading with memory-efficient two-stage sampling:
    1. Stage 1: Sample a dataset according to dataset-level weights (based on robot_type_sampling_probs)
    2. Stage 2: Uniformly sample a sample from the selected dataset
    
    Args:
        robot_type_sampling_probs: Optional dictionary mapping robot_type to sampling probability.
            Higher values increase the sampling frequency of that robot type.
            Example: {"S1-stationary": 2.0, "agibot-go1": 1.0, "human": 0.5}
            This will sample S1-stationary 2x more frequently than agibot-go1, and human 0.5x.
        num_samples_per_epoch: Number of samples to yield per epoch. If None, yields indefinitely.
    
    Usage example:
        # Create dataset with robot type sampling
        dataset = MultiCustomLeRobotDataset(
            repo_ids=["repo1", "repo2"],
            robot_type_sampling_probs={"S1-stationary": 2.0, "agibot-go1": 1.0},
            num_samples_per_epoch=100000
        )
        
        # Get distribution of robot types
        print(dataset.get_robot_type_distribution())
        
        # Use with DataLoader (no sampler needed for IterableDataset)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(
        self,
        repo_ids: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        robot_type_sampling_probs: dict[str, float] | None = None,
        parallel_load: bool = True,
        max_workers: int = 16,
        num_samples_per_epoch: int | None = None,
        seed: int = 233,
        static_action_threshold: float = 1e-3,
        static_chunk_max_fraction: float = 0.5,
        max_static_resample_attempts: int = 32,
    ):
        self.repo_ids = repo_ids
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        
        def _init_dataset(repo_id: str) -> tuple[str, CustomLeRobotDataset]:
            """Initialize a single dataset and return (repo_id, dataset) tuple."""
            try:
                repo_delta_timestamps = _filter_delta_timestamps_for_repo(delta_timestamps, repo_id)
                repo_episodes = episodes[repo_id] if episodes else None
                if is_droid_repo(repo_id):
                    available_episodes = get_available_droid_episodes(repo_id, repo_delta_timestamps)
                    if repo_episodes is not None:
                        available = set(available_episodes)
                        repo_episodes = [episode for episode in repo_episodes if episode in available]
                    else:
                        repo_episodes = available_episodes
                    if not repo_episodes:
                        raise RuntimeError(f"No generated DROID episodes with non-empty language found in {repo_id}")
                    print(f"DROID available episodes after filtering: {len(repo_episodes)}")
                dataset = CustomLeRobotDataset(
                    repo_id.split("/")[-1],
                    root=repo_id,
                    local_files_only=True,
                    episodes=repo_episodes,
                    image_transforms=image_transforms,
                    delta_timestamps=repo_delta_timestamps,
                    tolerance_s=tolerances_s[repo_id],  # Please make sure the error is acceptable by given appropriate delta_timestamps
                    download_videos=download_videos,
                    video_backend=video_backend,
                )
                return repo_id, dataset
            except:
                print(f"✗ Failed to load dataset {repo_id}")
                return repo_id, None
        
        # Construct the underlying datasets with optional parallel loading
        if parallel_load and len(repo_ids) > 1:
            print(f"Parallel loading {len(repo_ids)} datasets with max_workers={max_workers}...")
            datasets_dict = {}
            failed_repos = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_repo = {executor.submit(_init_dataset, repo_id): repo_id 
                                  for repo_id in repo_ids}
                for future in as_completed(future_to_repo):
                    repo_id = future_to_repo[future]
                    result_repo_id, dataset = future.result()
                    if dataset is None:
                        failed_repos.append(result_repo_id)
                        continue
                    datasets_dict[result_repo_id] = dataset
                    print(f"✓ Loaded dataset: {result_repo_id}")
            
            # Maintain original order of datasets, only include successfully loaded ones
            self._datasets = []
            self.successful_repo_ids = []
            for repo_id in repo_ids:
                if repo_id in datasets_dict:
                    self._datasets.append(datasets_dict[repo_id])
                    self.successful_repo_ids.append(repo_id)
            
            print(f"Successfully loaded {len(self._datasets)}/{len(repo_ids)} datasets")
            if failed_repos:
                print(f"Failed to load {len(failed_repos)} datasets: {failed_repos}")
        else:
            print(f"Sequential loading {len(repo_ids)} datasets...")
            self._datasets = []
            self.successful_repo_ids = []
            for repo_id in repo_ids:
                _, dataset = _init_dataset(repo_id)
                if dataset is None:
                    continue
                self._datasets.append(dataset)
                self.successful_repo_ids.append(repo_id)
                print(f"✓ Loaded dataset: {repo_id}")

        # Check if we have at least one dataset loaded
        if len(self._datasets) == 0:
            raise RuntimeError("No datasets were successfully loaded!")
        
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.successful_repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)
        self.kept_keys = {
            "cartesian_so3_dict.cartesian_pose_command",
            "cartesian_so3_dict.cartesian_pose_state",
            "images_dict.head.rgb",
            "images_dict.right.rgb",
            "images_dict.left.rgb",
            "task",
            "subtask",
        }

        self.robot_id_mapping = ROBOT_ID_MAPPING
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.robot_type_sampling_probs = robot_type_sampling_probs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.seed = seed
        self.static_action_threshold = static_action_threshold
        self.static_chunk_max_fraction = static_chunk_max_fraction
        self.max_static_resample_attempts = max_static_resample_attempts
        self.preprocess_to_canonical = any(is_droid_repo(repo_id) for repo_id in self.successful_repo_ids)
        self.astribot_pipeline = AstribotPipeline()
        self.droid_pipeline = DroidPipeline()
        
        # Calculate dataset-level information for efficient sampling
        print(f"robot_type_sampling_probs: {self.robot_type_sampling_probs}")
        self._dataset_weights = self._compute_dataset_weights()
        
        robot_type_distribution = self.get_robot_type_distribution()
        print(f"robot_type_distribution:\n{robot_type_distribution}")
        print(f"num_samples_per_epoch: {self.num_samples_per_epoch}")
        
        # Synchronize all processes in distributed training to avoid race conditions
        # This ensures all ranks finish loading datasets before any rank proceeds
        if dist.is_available() and dist.is_initialized():
            print(f"[Rank {dist.get_rank()}] Waiting for all ranks to finish loading datasets...")
            dist.barrier()
            print(f"[Rank {dist.get_rank()}] All ranks finished loading datasets, proceeding...")

    @staticmethod
    def _add_batch_dim_for_pipeline(item: dict) -> dict:
        batched = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and key not in {
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
                "dataset_index",
                "robot_id",
            }:
                batched[key] = value.unsqueeze(0)
            else:
                batched[key] = value
        if isinstance(batched.get("robot_id"), int):
            batched["robot_id"] = torch.tensor([batched["robot_id"]], dtype=torch.long)
        elif isinstance(batched.get("robot_id"), torch.Tensor) and batched["robot_id"].ndim == 0:
            batched["robot_id"] = batched["robot_id"].view(1)
        if isinstance(batched.get("arm_layout"), str):
            batched["arm_layout"] = [batched["arm_layout"]]
        return batched

    @staticmethod
    def _canonical_sample(processed: dict, source: dict) -> dict:
        keys = [
            "action",
            "action_static_reference",
            "state",
            "arm_mask",
            "observation.head",
            "observation.right",
            "task",
            "subtask",
            "dataset_index",
            "dataset_name",
            "robot_type",
            "robot_id",
            "arm_layout",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
        ]
        sample = {}
        for key in keys:
            if key in processed:
                value = processed[key]
            elif key in source:
                value = source[key]
            else:
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1:
                value = value.squeeze(0)
            sample[key] = value

        if "observation.head" in sample and "observation.right" not in sample:
            sample["observation.right"] = torch.zeros_like(sample["observation.head"])
        return sample

    def _is_static_chunk(self, sample: dict) -> bool:
        action = sample.get("action_static_reference", sample.get("action"))
        if not isinstance(action, torch.Tensor) or action.numel() == 0:
            return False

        if action.ndim > 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        if action.ndim < 2 or action.shape[-1] % 7 != 0:
            return False

        num_arms = action.shape[-1] // 7
        action = action.reshape(*action.shape[:-1], num_arms, 7)
        pose_norm = torch.linalg.vector_norm(action[..., :6], dim=-1)

        arm_mask = sample.get("arm_mask")
        if isinstance(arm_mask, torch.Tensor):
            if arm_mask.ndim > 1 and arm_mask.shape[0] == 1:
                arm_mask = arm_mask.squeeze(0)
            if arm_mask.ndim == 1 and arm_mask.numel() == num_arms:
                active_mask = arm_mask.to(device=pose_norm.device, dtype=torch.bool)
                pose_norm = pose_norm[..., active_mask]

        if pose_norm.numel() == 0:
            return False

        static_fraction = (pose_norm <= self.static_action_threshold).to(torch.float32).mean()
        return bool(static_fraction >= self.static_chunk_max_fraction)

    def _preprocess_canonical_item(self, item: dict, dataset_idx: int, dataset: CustomLeRobotDataset, rng: random.Random) -> dict:
        dataset_name = self.successful_repo_ids[dataset_idx]
        dataset_name = dataset_name.split("/")[-2] + "/" + dataset_name.split("/")[-1]
        robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)
        item = dict(item)
        item.update({
            "dataset_index": torch.tensor(dataset_idx),
            "dataset_name": dataset_name,
            "robot_type": robot_type,
            "robot_id": self.robot_id_mapping[robot_type],
            "arm_layout": ARM_LAYOUT_BY_ROBOT_TYPE.get(robot_type, "dual"),
        })

        if robot_type == "droid":
            available_external_keys = [key for key in DROID_EXTERNAL_IMAGE_KEYS if key in item]
            if available_external_keys:
                selected_key = rng.choice(available_external_keys)
                item["observation.images.droid_external"] = item[selected_key]
            if DROID_WRIST_IMAGE_KEY not in item and "observation.images.droid_external" in item:
                item[DROID_WRIST_IMAGE_KEY] = torch.zeros_like(item["observation.images.droid_external"])
            processed = self.droid_pipeline(self._add_batch_dim_for_pipeline(item))
        else:
            processed = self.astribot_pipeline(self._add_batch_dim_for_pipeline(item))
        return self._canonical_sample(processed, item)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Iterator that yields samples using two-stage sampling strategy.
        Handles distributed training by splitting work across workers.
        """
        # Get worker info for multi-worker DataLoader
        worker_info = torch.utils.data.get_worker_info()
        
        # Get distributed training info
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        # Calculate unique seed for this worker and rank
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Each worker gets a unique seed
            worker_seed = self.seed + rank * 1000 + worker_id
        else:
            worker_id = 0
            num_workers = 1
            worker_seed = self.seed + rank
        
        # Setup random generators
        rng = random.Random(worker_seed)
        torch_rng = torch.Generator()
        torch_rng.manual_seed(worker_seed)
        
        # Normalize weights for random.choices
        total_weight = sum(self._dataset_weights)
        normalized_weights = [w / total_weight for w in self._dataset_weights]
        
        # Calculate how many samples this worker should yield
        if self.num_samples_per_epoch is not None:
            # Distribute samples across world_size and num_workers
            total_samples_per_worker_rank = self.num_samples_per_epoch // (world_size * num_workers)
            # Assign remaining samples to first workers
            if rank * num_workers + worker_id < self.num_samples_per_epoch % (world_size * num_workers):
                total_samples_per_worker_rank += 1
            samples_to_yield = total_samples_per_worker_rank
        else:
            # Yield indefinitely if num_samples_per_epoch is None
            samples_to_yield = None

        sample_count = 0
        while samples_to_yield is None or sample_count < samples_to_yield:
            resample_attempts = 0
            while True:
                # Stage 1: Sample a dataset according to weights
                dataset_idx = rng.choices(range(len(self._datasets)), weights=normalized_weights, k=1)[0]
                dataset = self._datasets[dataset_idx]

                # Stage 2: Uniformly sample an index from the selected dataset
                sample_idx = rng.randint(0, len(dataset) - 1)
                robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)
                if robot_type != "human":
                    sample_idx = sample_idx // 8 * 8  # Downsample to reduce redundancy

                # Get the item from the dataset
                item = dataset[sample_idx]

                if self.preprocess_to_canonical:
                    sample = self._preprocess_canonical_item(item, dataset_idx, dataset, rng)
                    if (
                        self._is_static_chunk(sample)
                        and resample_attempts < self.max_static_resample_attempts
                    ):
                        resample_attempts += 1
                        continue

                    sample.pop("action_static_reference", None)
                    yield sample
                    sample_count += 1
                    break

                # Filter and add metadata
                ret_dict = {}
                for key in self.kept_keys:
                    if key in item:
                        ret_dict[key] = item[key]

                dataset_name = self.successful_repo_ids[dataset_idx]
                dataset_name = dataset_name.split("/")[-2] + "/" + dataset_name.split("/")[-1]
                robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)

                ret_dict.update({
                    "dataset_index": torch.tensor(dataset_idx),
                    "dataset_name": dataset_name,
                    "robot_type": robot_type,
                    "robot_id": self.robot_id_mapping[robot_type],
                    "arm_layout": ARM_LAYOUT_BY_ROBOT_TYPE.get(robot_type, "dual"),
                })

                if (
                    self._is_static_chunk(ret_dict)
                    and resample_attempts < self.max_static_resample_attempts
                ):
                    resample_attempts += 1
                    continue

                ret_dict.pop("action_static_reference", None)
                yield ret_dict
                sample_count += 1
                break

    def _compute_dataset_weights(self) -> List[float]:
        """
        Compute sampling weights for each dataset based on robot_type_sampling_probs.
        
        Returns:
            List of sampling weights for each dataset
        """
        dataset_weights = []
        
        for dataset in self._datasets:
            # Get weight for this dataset's robot_type
            if self.robot_type_sampling_probs is not None:
                robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)
                weight = self.robot_type_sampling_probs.get(robot_type, 1.0)
            else:
                weight = 1.0
            
            dataset_weights.append(weight * len(dataset))
        
        return dataset_weights
    
    def get_robot_type_distribution(self) -> dict[str, int]:
        """
        Get the distribution of robot types in the dataset.
        
        Returns:
            A dictionary mapping robot_type to the number of samples.
        """
        distribution = {}
        for dataset in self._datasets:
            robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)
            if robot_type not in distribution:
                distribution[robot_type] = 0
            distribution[robot_type] += dataset.num_frames
        return distribution

    def __repr__(self):
        total_frames = sum(ds.num_frames for ds in self._datasets)
        total_episodes = sum(ds.num_episodes for ds in self._datasets)
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs (requested): {len(self.repo_ids)},\n"
            f"  Successfully Loaded: {len(self.successful_repo_ids)},\n"
            f"  Loaded IDs: {self.successful_repo_ids},\n"
            f"  Total Frames: {total_frames},\n"
            f"  Total Episodes: {total_episodes},\n"
            f"  Samples Per Epoch: {self.num_samples_per_epoch if self.num_samples_per_epoch else 'Infinite'},\n"
            f"  Robot Type Sampling Probs: {self.robot_type_sampling_probs},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )


class MultiCustomLeRobotDatasetFromYAML(Dataset):
    """A Dataset consisting of multiple underlying `CustomLeRobotDataset`s loaded from YAML config.
    
    This class loads multiple datasets based on a YAML configuration file and concatenates them
    into a single Dataset using PyTorch's default sampling (no custom sampling weights).
    
    YAML Format:
        datasets:
          - repo_id: path/to/dataset1
            max_episodes: 100  # Load first 100 episodes
            tolerance_s: 0.0001  # Optional: tolerance for timestamp matching
          - repo_id: path/to/dataset2
            # No max_episodes: load all episodes
            tolerance_s: 0.001
    
    Args:
        yaml_path: Path to the YAML configuration file
        image_transforms: Optional image transformation function to apply
        delta_timestamps: Dictionary of delta timestamps for temporal queries
        download_videos: Whether to download video files
        video_backend: Video decoding backend to use
        parallel_load: Whether to load datasets in parallel
        max_workers: Maximum number of worker threads for parallel loading
    
    Usage example:
        # Create dataset from YAML config
        dataset = MultiCustomLeRobotDatasetFromYAML(
            yaml_path="config/datasets.yaml",
            image_transforms=transforms,
            delta_timestamps={"images_dict.head.rgb": [0.0, 0.033, 0.066]},
        )
        
        # Use with standard DataLoader (supports any Sampler)
        from torch.utils.data import DataLoader, RandomSampler
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    """

    def __init__(
        self,
        yaml_path: str,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        parallel_load: bool = True,
        max_workers: int = 16,
    ):
        # Load YAML configuration
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'datasets' not in config:
            raise ValueError(f"YAML config must contain 'datasets' key. Found keys: {list(config.keys())}")
        
        dataset_configs = config['datasets']
        if not isinstance(dataset_configs, list) or len(dataset_configs) == 0:
            raise ValueError(f"'datasets' must be a non-empty list in YAML config")
        
        # Extract repo_ids, episodes, and tolerances from config
        self.repo_ids = []
        episodes_dict = {}
        tolerances_dict = {}
        
        for ds_config in dataset_configs:
            if 'repo_id' not in ds_config:
                raise ValueError(f"Each dataset config must contain 'repo_id'. Found: {ds_config}")
            
            repo_id = ds_config['repo_id']
            self.repo_ids.append(repo_id)
            
            # Handle max_episodes - if specified, create list of episode indices
            if 'max_episodes' in ds_config:
                max_episodes = ds_config['max_episodes']
                if max_episodes is not None and max_episodes > 0:
                    episodes_dict[repo_id] = list(range(max_episodes))
                else:
                    episodes_dict[repo_id] = None
            else:
                episodes_dict[repo_id] = None
            
            # Handle tolerance_s
            tolerances_dict[repo_id] = ds_config.get('tolerance_s', 0.0001)
        
        print(f"Loading {len(self.repo_ids)} datasets from YAML config: {yaml_path}")
        for repo_id in self.repo_ids:
            max_eps = len(episodes_dict[repo_id]) if episodes_dict[repo_id] else "all"
            print(f"  - {repo_id}: max_episodes={max_eps}, tolerance_s={tolerances_dict[repo_id]}")
        
        # Initialize function for a single dataset
        def _init_dataset(repo_id: str) -> tuple[str, CustomLeRobotDataset]:
            """Initialize a single dataset and return (repo_id, dataset) tuple."""
            try:
                repo_delta_timestamps = _filter_delta_timestamps_for_repo(delta_timestamps, repo_id)
                repo_episodes = episodes_dict[repo_id]
                if is_droid_repo(repo_id):
                    available_episodes = get_available_droid_episodes(repo_id, repo_delta_timestamps)
                    if repo_episodes is not None:
                        available = set(available_episodes)
                        repo_episodes = [episode for episode in repo_episodes if episode in available]
                    else:
                        repo_episodes = available_episodes
                    if not repo_episodes:
                        raise RuntimeError(f"No generated DROID episodes with non-empty language found in {repo_id}")
                    print(f"DROID available episodes after filtering: {len(repo_episodes)}")
                dataset = CustomLeRobotDataset(
                    repo_id.split("/")[-1],
                    root=repo_id,
                    local_files_only=True,
                    episodes=repo_episodes,
                    image_transforms=image_transforms,
                    delta_timestamps=repo_delta_timestamps,
                    tolerance_s=tolerances_dict[repo_id],
                    download_videos=download_videos,
                    video_backend=video_backend,
                )
                return repo_id, dataset
            except Exception as e:
                print(f"✗ Failed to load dataset {repo_id}: {e}")
                return repo_id, None
        
        # Load datasets with optional parallel loading
        if parallel_load and len(self.repo_ids) > 1:
            print(f"Parallel loading {len(self.repo_ids)} datasets with max_workers={max_workers}...")
            datasets_dict = {}
            failed_repos = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_repo = {executor.submit(_init_dataset, repo_id): repo_id 
                                  for repo_id in self.repo_ids}
                for future in as_completed(future_to_repo):
                    repo_id = future_to_repo[future]
                    result_repo_id, dataset = future.result()
                    if dataset is None:
                        failed_repos.append(result_repo_id)
                        continue
                    datasets_dict[result_repo_id] = dataset
                    print(f"✓ Loaded dataset: {result_repo_id} ({len(dataset)} samples)")
            
            # Maintain original order of datasets
            self._datasets = []
            self.successful_repo_ids = []
            for repo_id in self.repo_ids:
                if repo_id in datasets_dict:
                    self._datasets.append(datasets_dict[repo_id])
                    self.successful_repo_ids.append(repo_id)
            
            print(f"Successfully loaded {len(self._datasets)}/{len(self.repo_ids)} datasets")
            if failed_repos:
                print(f"Failed to load {len(failed_repos)} datasets: {failed_repos}")
        else:
            print(f"Sequential loading {len(self.repo_ids)} datasets...")
            self._datasets = []
            self.successful_repo_ids = []
            for repo_id in self.repo_ids:
                _, dataset = _init_dataset(repo_id)
                if dataset is None:
                    continue
                self._datasets.append(dataset)
                self.successful_repo_ids.append(repo_id)
                print(f"✓ Loaded dataset: {repo_id} ({len(dataset)} samples)")
        
        # Check if we have at least one dataset loaded
        if len(self._datasets) == 0:
            raise RuntimeError("No datasets were successfully loaded!")
        
        # Calculate cumulative lengths for efficient indexing
        self._cumulative_lengths = []
        cumsum = 0
        for ds in self._datasets:
            cumsum += len(ds)
            self._cumulative_lengths.append(cumsum)
        
        self._total_length = cumsum
        
        # Disable any data keys that are not common across all datasets
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        
        for repo_id, ds in zip(self.successful_repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
            self.disabled_features.update(extra_keys)
        
        self.robot_id_mapping = ROBOT_ID_MAPPING
        
        # Print dataset statistics
        robot_type_distribution = self.get_robot_type_distribution()
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {self._total_length}")
        print(f"  Total episodes: {sum(ds.num_episodes for ds in self._datasets)}")
        print(f"  Robot type distribution: {robot_type_distribution}")
        
        # Synchronize all processes in distributed training
        if dist.is_available() and dist.is_initialized():
            print(f"[Rank {dist.get_rank()}] Waiting for all ranks to finish loading datasets...")
            dist.barrier()
            print(f"[Rank {dist.get_rank()}] All ranks finished loading datasets, proceeding...")
    
    def __len__(self) -> int:
        """Return the total number of samples across all datasets."""
        return self._total_length
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the concatenated datasets.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            Dictionary containing the sample data with common keys
        """
        if idx < 0 or idx >= self._total_length:
            raise IndexError(f"Index {idx} out of range [0, {self._total_length})")
        
        # Find which dataset this index belongs to using binary search
        dataset_idx = 0
        for i, cumlen in enumerate(self._cumulative_lengths):
            if idx < cumlen:
                dataset_idx = i
                break
        
        # Calculate the local index within the selected dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self._cumulative_lengths[dataset_idx - 1]
        
        # Get the item from the selected dataset
        dataset = self._datasets[dataset_idx]
        item = dataset[local_idx]
        
        # Filter to keep only common keys
        ret_dict = item
        
        # Add dataset metadata
        dataset_name = self.successful_repo_ids[dataset_idx]
        dataset_name = dataset_name.split("/")[-2] + "/" + dataset_name.split("/")[-1]
        robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)
        
        ret_dict.update({
            "dataset_index": torch.tensor(dataset_idx),
            "dataset_name": dataset_name,
            "robot_type": robot_type,
            "robot_id": self.robot_id_mapping[robot_type],
            "arm_layout": ARM_LAYOUT_BY_ROBOT_TYPE.get(robot_type, "dual"),
        })
        
        return ret_dict
    
    def get_robot_type_distribution(self) -> dict[str, int]:
        """
        Get the distribution of robot types in the dataset.
        
        Returns:
            A dictionary mapping robot_type to the number of samples.
        """
        distribution = {}
        for dataset in self._datasets:
            robot_type = robot_type_for_dataset(dataset.root, dataset.meta.robot_type)
            if robot_type not in distribution:
                distribution[robot_type] = 0
            distribution[robot_type] += dataset.num_frames
        return distribution
    
    def __repr__(self):
        total_frames = sum(ds.num_frames for ds in self._datasets)
        total_episodes = sum(ds.num_episodes for ds in self._datasets)
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs (requested): {len(self.repo_ids)},\n"
            f"  Successfully Loaded: {len(self.successful_repo_ids)},\n"
            f"  Loaded IDs: {self.successful_repo_ids},\n"
            f"  Total Frames: {total_frames},\n"
            f"  Total Episodes: {total_episodes},\n"
            f"  Total Samples: {self._total_length},\n"
            f")"
        )
