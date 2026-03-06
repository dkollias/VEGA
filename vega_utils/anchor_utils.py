"""Anchor feature cache and extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from model import get_clip_visual_features_batch
from vega_utils.common import emotion_labels, list_image_file_abs_path_recursive


def _get_anchor_cache_path(args) -> Path:
    return Path("anchor") / f"{args.expr_img_folder}_anchor.pt"


def _get_fallback_anchor_cache_path(args) -> Path:
    cache_root = Path(args.expr_img_root).parent / "anchor_dicts"
    return cache_root / f"{args.expr_img_folder}.pt"


def _resolve_anchor_image_root(args) -> Path:
    configured = Path(str(getattr(args, "expr_img_root", "")))
    canonical = Path("anchor") / str(args.expr_img_folder)
    suffixed = Path("anchor") / f"{args.expr_img_folder}_anchor"

    candidates = []
    for path in (configured, canonical, suffixed):
        if str(path) not in [str(p) for p in candidates]:
            candidates.append(path)

    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    return configured


def _save_anchor_cache(cache_path: Path, anchor_dict: Dict) -> None:
    anchor_img_dict_cpu = {}
    for label, info in anchor_dict["anchor_img_dict"].items():
        feature = info.get("feature")
        anchor_img_dict_cpu[label] = {
            "feature": feature.detach().cpu() if isinstance(feature, torch.Tensor) else feature,
        }

    anchor_dict_cpu = {
        "anchor_center": (
            anchor_dict["anchor_center"].detach().cpu()
            if isinstance(anchor_dict["anchor_center"], torch.Tensor)
            else anchor_dict["anchor_center"]
        ),
        "anchor_img_dict": anchor_img_dict_cpu,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchor_dict_cpu, cache_path)


def _load_anchor_cache(cache_path: Path, device: torch.device) -> Optional[Dict]:
    if not cache_path.exists():
        return None
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        cached = torch.load(cache_path, map_location="cpu")

    if not isinstance(cached, dict):
        return None
    if "anchor_center" not in cached or "anchor_img_dict" not in cached:
        return None

    if isinstance(cached["anchor_center"], torch.Tensor):
        cached["anchor_center"] = cached["anchor_center"].to(device)

    needs_resave = False
    normalized_anchor_img_dict = {}
    for label, info in cached["anchor_img_dict"].items():
        feature = info.get("feature")
        if isinstance(feature, torch.Tensor):
            feature = feature.to(device)
        normalized_anchor_img_dict[label] = {"feature": feature}
        if set(info.keys()) != {"feature"}:
            needs_resave = True

    cached["anchor_img_dict"] = normalized_anchor_img_dict
    if needs_resave:
        _save_anchor_cache(cache_path, cached)
    return cached


def get_anchors(args) -> Dict:
    """Load cached anchors or build anchor features from image folders."""
    device = torch.device("cuda" if args.cuda else "cpu")
    cache_path = _get_anchor_cache_path(args)
    cached = _load_anchor_cache(cache_path, device)
    if cached is not None:
        return cached

    fallback_cache_path = _get_fallback_anchor_cache_path(args)
    fallback_cached = _load_anchor_cache(fallback_cache_path, device)
    if fallback_cached is not None:
        _save_anchor_cache(cache_path, fallback_cached)
        return fallback_cached

    image_root = _resolve_anchor_image_root(args)
    if not image_root.exists():
        raise FileNotFoundError(
            f"Anchor image directory not found: {image_root}. "
            f"Please provide anchor images under 'anchor/{args.expr_img_folder}' "
            f"or 'anchor/{args.expr_img_folder}_anchor', or provide anchor cache "
            f"at '{cache_path}'."
        )

    image_files = list_image_file_abs_path_recursive(str(image_root))
    if not image_files:
        raise ValueError(
            f"No image files found under anchor directory: {image_root}. "
            f"Supported format should include .jpg/.jpeg/.png."
        )

    labels = emotion_labels[args.Dataset]
    image_files_by_label = {label: [] for label in labels}

    for file_path in image_files:
        file_str = str(file_path)
        for label in labels:
            if label in file_str:
                image_files_by_label[label].append(file_path)
                break

    empty_labels = [label for label, files in image_files_by_label.items() if len(files) == 0]
    if empty_labels:
        raise ValueError(
            f"Missing anchor images for labels: {empty_labels}. "
            f"Current root: {image_root}"
        )

    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(args.CLIP_Model).cuda()
    clip_processor = CLIPProcessor.from_pretrained(args.CLIP_Model)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    anchor_center = []
    anchor_img_dict = {}
    with torch.no_grad():
        for label in labels:
            label_image_files = image_files_by_label[label]
            anchor_img_dict[label] = {
                "feature": get_clip_visual_features_batch(
                    label_image_files,
                    clip_model,
                    clip_processor,
                )
            }
            center_feature = anchor_img_dict[label]["feature"].mean(dim=0)
            anchor_center.append(center_feature.unsqueeze(0))

    anchor_center = torch.cat(anchor_center)
    anchor_dict = {"anchor_center": anchor_center, "anchor_img_dict": anchor_img_dict}
    _save_anchor_cache(cache_path, anchor_dict)
    return anchor_dict
