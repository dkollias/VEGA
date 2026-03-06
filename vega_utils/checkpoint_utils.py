"""Checkpoint save and rotation utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import torch


def remove_file_from_dir_contain_pattern(
    dir_root: Union[str, Path],
    pattern: re.Pattern,
    value: float,
) -> bool:
    """
    Delete files in ``dir_root`` whose metric value in filename is lower than ``value``.

    Returns:
        bool:
            - ``True`` when a new checkpoint should be saved.
            - ``False`` when existing files are all better and save can be skipped.
    """
    dir_root = Path(dir_root)
    file_list = list(dir_root.glob("*"))
    should_save = False
    has_existing_metric_file = False

    for file_path in file_list:
        match = re.search(pattern, str(file_path))
        if not match:
            continue

        has_existing_metric_file = True
        raw_val = match.group(1).strip("[]")
        try:
            existing_metric_value = float(raw_val)
        except ValueError:
            continue

        if value > existing_metric_value:
            file_path.unlink()
            print(f"Deleted file: {file_path}")
            should_save = True

    return should_save or not has_existing_metric_file


def _is_cls_transformer_key(key: str) -> bool:
    """Whitelist parameters that belong to backbone (CLS) branch."""
    cls_prefixes = (
        "speaker_embeddings",
        "t_t",
        "a_t",
        "v_t",
        "a_a",
        "t_a",
        "v_a",
        "v_v",
        "t_v",
        "a_v",
        "t_t_gate",
        "a_t_gate",
        "v_t_gate",
        "a_a_gate",
        "t_a_gate",
        "v_a_gate",
        "v_v_gate",
        "t_v_gate",
        "a_v_gate",
        "features_reduce_t",
        "features_reduce_a",
        "features_reduce_v",
        "last_gate",
        "textf_input",
        "acouf_input",
        "visuf_input",
        "t_output_layer",
        "a_output_layer",
        "v_output_layer",
        "all_output_layer",
        "a_cls_temp",
        "v_cls_temp",
        "t_cls_temp",
    )
    return key.startswith(cls_prefixes)


def save_ckp(
    ckp_root: Union[str, Path],
    pattern: str,
    ckp_path: Path,
    model: torch.nn.Module,
    value: float,
) -> Optional[dict]:
    """Save filtered backbone state_dict only if current score is better."""
    should_save = remove_file_from_dir_contain_pattern(ckp_root, re.compile(pattern), value)
    if not should_save:
        return None

    full_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v.detach().cpu().clone()
        for k, v in full_state_dict.items()
        if _is_cls_transformer_key(k)
    }
    torch.save(filtered_state_dict, ckp_path)

    return {"path": ckp_path, "value": value}


def save_best_checkpoint(
    args,
    epoch: int,
    model: torch.nn.Module,
    value: float,
    metric: str = "f1",
) -> Optional[dict]:
    """Create checkpoint paths/pattern then save best checkpoint."""
    ckp_path = args.checkpoint_root / f"BEST_{metric}_[{value}]_epoch{epoch}.pth"
    pattern = rf"BEST_{metric}_\[(.*?)\]_epoch.*\.pth$"
    ckp_path.parent.mkdir(parents=True, exist_ok=True)

    return save_ckp(
        ckp_root=ckp_path.parent,
        pattern=pattern,
        ckp_path=ckp_path,
        model=model,
        value=value,
    )
