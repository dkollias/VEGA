import argparse
import contextlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score
from tqdm import tqdm

from configs.iemocap_config import IEMOCAP_CONFIG
from main import create_model, setup_data_and_loss
from vega_utils.anchor_utils import get_anchors


def _print_section(title: str):
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(ckp):
    if not isinstance(ckp, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckp)}")

    if "state_dict" in ckp and isinstance(ckp["state_dict"], dict):
        return ckp["state_dict"]
    return ckp


def _build_runtime_args(cli_args):
    args = Namespace()
    args.Dataset = cli_args.dataset
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.cuda = torch.cuda.is_available() and not cli_args.cpu

    args.audio_dim = 1582 if args.Dataset == "IEMOCAP" else 300
    args.visual_dim = 342
    args.text_dim = 1024
    args.n_speakers = 9 if args.Dataset == "MELD" else 2
    args.n_classes = 7 if args.Dataset == "MELD" else 6

    args.hidden_dim = cli_args.hidden_dim
    args.n_head = cli_args.n_head
    args.dropout = cli_args.dropout

    args.outlayer_drop = cli_args.outlayer_drop
    args.outlayer_num = cli_args.outlayer_num
    args.outlayer_activation_fn = cli_args.outlayer_activation_fn

    args.clip_loss = cli_args.clip_loss
    args.clip_dim = cli_args.clip_dim
    args.clip_proj_layer_num = cli_args.clip_proj_layer_num
    args.clip_proj_activation_fn = cli_args.clip_proj_activation_fn
    args.clip_proj_drop = cli_args.clip_proj_drop

    args.cls_loss = True
    args.rand = cli_args.rand
    args.expr_img_folder = cli_args.expr_img_folder
    expr_img_root = Path("anchor") / str(args.expr_img_folder)
    fallback_expr_img_root = Path("anchor") / f"{args.expr_img_folder}_anchor"
    if not expr_img_root.exists() and fallback_expr_img_root.exists():
        expr_img_root = fallback_expr_img_root
    args.expr_img_root = str(expr_img_root)
    return args


def _evaluate_cls_f1(args, model, test_loader, anchor_dict):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="CLS inference"):
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            textf = textf.permute(1, 2, 0)
            acouf = acouf.permute(1, 2, 0)
            visuf = visuf.permute(1, 2, 0)

            with (torch.amp.autocast("cuda") if args.cuda else contextlib.nullcontext()):
                _, _, _, all_logit, _ = model.forward_backbone(
                    textf, visuf, acouf, umask, qmask, lengths
                )
                all_logit = all_logit.view(-1, all_logit.size(2))

            labels_ = label.view(-1)
            preds.append(torch.argmax(all_logit, dim=1).cpu().numpy())
            labels.append(labels_.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    valid = labels != -1

    f1 = round(f1_score(labels[valid], preds[valid], average="weighted") * 100, 2)
    return f1, labels[valid], preds[valid]


def main():
    cfg = IEMOCAP_CONFIG
    parser = argparse.ArgumentParser(description="checkpoint inference")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint/IEMOCAP.pth",
        help="Path to .pth checkpoint (default: checkpoint/IEMOCAP.pth)",
    )
    parser.add_argument("--dataset", choices=["IEMOCAP", "MELD"], default=cfg["Dataset"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=cfg["batch_size"], help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=cfg["num_workers"], help="DataLoader num_workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # Backbone/model args (manual input, no checkpoint args dependency)
    parser.add_argument("--hidden_dim", type=int, default=cfg["hidden_dim"], help="Backbone hidden dimension")
    parser.add_argument("--n_head", type=int, default=cfg["n_head"], help="Backbone attention heads")
    parser.add_argument("--dropout", type=float, default=cfg["dropout"], help="Backbone dropout")
    parser.add_argument("--outlayer_drop", type=float, default=cfg["outlayer_drop"], help="Classifier head dropout")
    parser.add_argument("--outlayer_num", type=int, default=cfg["outlayer_num"], help="Classifier head layer count")
    parser.add_argument(
        "--outlayer_activation_fn",
        type=str,
        default=cfg["outlayer_activation_fn"],
        choices=["relu", "gelu", "silu", "tanh", "leaky_relu", "elu", "none"],
        help="Classifier head activation",
    )

    # Optional CLIP branch args
    parser.add_argument("--clip_loss", action="store_true", default=cfg["clip_loss"], help="Enable CLIP branch when building model")
    parser.add_argument("--clip_dim", type=int, default=cfg["clip_dim"], help="CLIP feature dimension")
    parser.add_argument("--clip_proj_layer_num", type=int, default=cfg["clip_proj_layer_num"], help="CLIP projection layer count")
    parser.add_argument(
        "--clip_proj_activation_fn",
        type=str,
        default=cfg["clip_proj_activation_fn"],
        choices=["relu", "gelu", "silu", "tanh", "leaky_relu", "elu", "none"],
        help="CLIP projection activation",
    )
    parser.add_argument("--clip_proj_drop", type=float, default=cfg["clip_proj_drop"], help="CLIP projection dropout")

    # Anchor settings
    parser.add_argument("--expr_img_folder", type=str, default=cfg["expr_img_folder"], help="Anchor folder under anchor/")
    parser.add_argument("--rand", type=float, default=cfg["rand"], help="Random anchor sampling ratio")
    args_cli = parser.parse_args()
    if not Path(args_cli.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")

    ckp = _load_checkpoint(args_cli.checkpoint)
    state_dict = _extract_state_dict(ckp)
    args = _build_runtime_args(args_cli)

    _print_section("Inference Setup")
    print(f"Checkpoint: {Path(args_cli.checkpoint).resolve()}")
    print(f"Device: {'GPU' if args.cuda else 'CPU'}")
    print(
        f"Dataset: {args.Dataset} | batch_size: {args.batch_size} | num_workers: {args.num_workers} | "
        f"clip_loss: {args.clip_loss}"
    )

    _, test_loader, _, _ = setup_data_and_loss(args)
    model = create_model(args)

    model_state = model.state_dict()
    loadable_state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    _, _ = model.load_state_dict(loadable_state_dict, strict=False)

    _print_section("Checkpoint Load")
    print(f"Total ckpt params: {len(state_dict)}")
    print(f"Loaded params: {len(loadable_state_dict)}")

    anchor_dict = get_anchors(args) if args.clip_loss else None
    cls_f1, labels, preds = _evaluate_cls_f1(args, model, test_loader, anchor_dict)
    acc = round(float(precision_score(labels, preds, average="weighted", zero_division=0) * 100), 2)

    _print_section("Evaluation")
    print(f"Weighted F1: {cls_f1:.2f}")
    print(f"Weighted ACC: {acc:.2f}")
    _print_section("Classification Report")
    print(classification_report(labels, preds, digits=4))


if __name__ == "__main__":
    main()
