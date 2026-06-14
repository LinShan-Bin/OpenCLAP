import os
import random
import logging
from typing import Any, List, Callable

import torch
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
import torchvision.transforms as transforms
from dataclasses import dataclass

from clap.custom_lerobot import (
    CustomLeRobotDataset,
    MultiCustomLeRobotDataset,
    MultiCustomLeRobotDatasetFromYAML,
    _filter_delta_timestamps_for_repo,
)


def set_global_seed(seed: int, get_worker_init_fn: bool = False):
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return default_worker_init_fn if get_worker_init_fn else None


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    """
    Safe worker init function that handles missing LOCAL_RANK for non-distributed training.
    Compatible with multi-machine training and IterableDataset.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    # LOCAL_RANK may not be set in non-distributed training, so default to 0
    global_rank = int(os.environ.get("LOCAL_RANK", "0"))
    process_seed = torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch
    base_seed = process_seed - worker_id

    # Create a seed sequence that mixes different "sources" and seeds every library
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 8,
            train_shuffle: bool = True,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = 1

        # shuffle unspecified for iteratable datasets
        # self.train_shuffle = train_shuffle
        # self.val_shuffle = val_shuffle

        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
            shuffle = None  # IterableDataset doesn't support shuffle
        else:
            worker_init_fn = self.worker_init_fn
            shuffle = True if self.train_sampler is None else None  # shuffle if no sampler provided
        
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )



from PIL import Image
import random

@dataclass
class random_crop_resize():
    def __init__(
        self,
        target_size=224
    ):
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image):
        width, height = image.size

        if width < height:
            crop_size = width
        else:
            crop_size = height

        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)

        image_cropped = image.crop((left, top, left + crop_size, top + crop_size))
        image_resized = image_cropped.resize((self.target_size, self.target_size), Image.BILINEAR)
        image_resized = self.to_tensor(image_resized)
        
        return image_resized



class LightningLerobot(LightningDataset):
    """
    This dataset samples video recorded using a random agent
    playing the gym environments defined in the Procgen Benchmark,
    see Cobbe et al. ICML (2020).
    """

    def __init__(
            self,
            data_root: list[str] | str = None,
            data_mix: list[str] | str = None,
            yaml_config_path: str = None,
            batch_size:int = 16,
            num_workers: int = 8,
            resolution: List[int] = [240, 320],
            fps: int = 30,
            num_frames: int = 17,
            delta_timestamps: dict[str, list[int]] = None,
            use_filter: bool = True,
            robot_type_sampling_probs: dict[str, float] = None,
            static_action_threshold: float = 1e-3,
            static_chunk_max_fraction: float = 0.5,
            max_static_resample_attempts: int = 32,
            test_repo_id: str = './data/astribot_pretrain/0901_pretrain_pnp_daxiong_3850_s1_10',
            test_episode: int = 0,
            **kwargs
    ) -> None:
        super(LightningLerobot, self).__init__(**kwargs)

        self.data_root_dir = data_root
        self.data_mix = data_mix
        self.yaml_config_path = yaml_config_path
        self.test_repo_id = test_repo_id
        self.test_episodes = [test_episode]
        self.batch_size = batch_size
        self.resolution = resolution
        self.fps = fps
        self.num_frames = num_frames
        if delta_timestamps is None:
            self.delta_timestamps = {
                'images_dict.head.rgb': [0],
                'images_dict.left.rgb': [0],
                'images_dict.right.rgb': [0],
                'cartesian_so3_dict.cartesian_pose_command': list(range(num_frames)),
                'cartesian_so3_dict.cartesian_pose_state': [0],
            }
        else:
            self.delta_timestamps = {
                k: list(v) for k, v in delta_timestamps.items()
            }
        
        self.use_filter = use_filter
        # if use_filter:
        #     # When filter_fc > 0, we apply low-pass filter to the cartesian pose command
        #     # Here we pad the cartesian pose command for the filter operation
        #     delta_t = 1 / fps
        #     if 'cartesian_so3_dict.cartesian_pose_command' in self.delta_timestamps:
        #         ts_action = self.delta_timestamps['cartesian_so3_dict.cartesian_pose_command']
        #         pad_length = len(ts_action)
        #         ts_before = np.linspace(ts_action[0] - pad_length * delta_t, ts_action[0], pad_length, endpoint=False)
        #         ts_after = np.linspace(ts_action[-1] + delta_t, ts_action[-1] + pad_length * delta_t, pad_length)
        #         ts_padded = np.concatenate([ts_before, ts_action, ts_after])
        #         self.delta_timestamps['cartesian_so3_dict.cartesian_pose_command'] = ts_padded.tolist()

        # self.tolerances_s = 0.4 / fps
        self.tolerances_s = 1
        self.robot_type_sampling_probs = robot_type_sampling_probs
        self.static_action_threshold = static_action_threshold
        self.static_chunk_max_fraction = static_chunk_max_fraction
        self.max_static_resample_attempts = max_static_resample_attempts

        self.num_workers = num_workers
        set_global_seed(42, get_worker_init_fn=False)
        self.worker_init_fn = default_worker_init_fn

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # Use MultiCustomLeRobotDatasetFromYAML if yaml_config_path is provided
            if self.yaml_config_path is not None:
                print(f"Using MultiCustomLeRobotDatasetFromYAML with config: {self.yaml_config_path}")
                self.train_dataset = MultiCustomLeRobotDatasetFromYAML(
                    yaml_path=self.yaml_config_path,
                    delta_timestamps=self.delta_timestamps,
                    image_transforms=transforms.Resize(self.resolution),
                )
            else:
                # Use the original MultiCustomLeRobotDataset with repo_ids
                print(f"Using MultiCustomLeRobotDataset with data_root and data_mix")
                
                # Get the repo_ids from the data_root_dir
                if isinstance(self.data_root_dir, str):
                    data_root_dirs = [self.data_root_dir]
                else:
                    data_root_dirs = self.data_root_dir
                assert isinstance(data_root_dirs, list), "data_root_dir must be a list or a string"
                
                repo_ids = []
                for data_root_dir in data_root_dirs:
                    if self.data_mix == "all":
                        dirs = os.listdir(data_root_dir)
                        abs_dirs = [os.path.join(data_root_dir, dir) for dir in dirs]
                        repo_ids += abs_dirs
                    elif isinstance(self.data_mix, list):
                        dirs = os.listdir(data_root_dir)
                        for dir in dirs:
                            if dir in self.data_mix:
                                repo_ids.append(os.path.join(data_root_dir, dir))
                    else:
                        raise ValueError(f"Invalid data_mix: {self.data_mix}")
                print(f"LeRobotDataset repo_ids: {repo_ids}")
                
                tolerances_s = {repo_id: self.tolerances_s for repo_id in repo_ids}
                
                self.train_dataset = MultiCustomLeRobotDataset(
                    repo_ids,
                    delta_timestamps=self.delta_timestamps,
                    image_transforms=transforms.Resize(self.resolution),
                    tolerances_s=tolerances_s,
                    robot_type_sampling_probs=self.robot_type_sampling_probs,
                    static_action_threshold=self.static_action_threshold,
                    static_chunk_max_fraction=self.static_chunk_max_fraction,
                    max_static_resample_attempts=self.max_static_resample_attempts,
                )
            
            val_delta_timestamps = _filter_delta_timestamps_for_repo(
                self.delta_timestamps, self.test_repo_id
            )
            self.val_dataset = CustomLeRobotDataset(
                self.test_repo_id.split("/")[-1],
                root=self.test_repo_id,
                local_files_only=True,
                episodes=self.test_episodes,
                delta_timestamps=val_delta_timestamps,
                image_transforms=transforms.Resize(self.resolution),
                tolerance_s=self.tolerances_s,
            )
        elif stage == "test":
            test_delta_timestamps = _filter_delta_timestamps_for_repo(
                self.delta_timestamps, self.test_repo_id
            )
            self.test_dataset = CustomLeRobotDataset(
                self.test_repo_id.split("/")[-1],
                root=self.test_repo_id,
                local_files_only=True,
                episodes=self.test_episodes,
                delta_timestamps=test_delta_timestamps,
                image_transforms=transforms.Resize(self.resolution),
                tolerance_s=self.tolerances_s,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")
