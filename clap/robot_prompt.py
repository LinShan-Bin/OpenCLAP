from __future__ import annotations

from numbers import Integral
from typing import List, Optional, Sequence, Union

from torch import Tensor


ROBOT_ID_TO_TYPE = {
    0: "S1-stationary",
    1: "agibot-go1",
    2: "human",
    3: "franka",
    4: "google_robot",
    5: "widowx",
    6: "droid",
}


RobotTypes = Optional[Union[str, bytes, int, Tensor, Sequence[Union[str, bytes, int, Tensor]]]]


def _as_list(value: RobotTypes) -> Optional[List[Union[str, int]]]:
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return [int(value.item())]
        return [int(v) for v in value.view(-1).tolist()]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, (str, int)):
        return [value]
    out = []
    for item in value:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        elif isinstance(item, Tensor):
            out.append(int(item.detach().cpu().item()))
        else:
            out.append(item)
    return out


def robot_type_name(value: Union[str, int]) -> str:
    if isinstance(value, Integral):
        robot_id = int(value)
        return ROBOT_ID_TO_TYPE.get(robot_id, f"robot_id_{robot_id}")
    return str(value)


def expand_robot_types(
    robot_types: RobotTypes,
    robot_ids: RobotTypes,
    batch_size: int,
    default: str = "unknown",
) -> List[str]:
    values = _as_list(robot_types)
    if values is None:
        values = _as_list(robot_ids)
    if values is None:
        values = [default]
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    if len(values) != batch_size:
        raise ValueError(f"robot_types length {len(values)} does not match batch size {batch_size}")
    return [robot_type_name(value) for value in values]


def robot_prompt_line(robot_type: str) -> str:
    return f"Robot category: {robot_type}"
