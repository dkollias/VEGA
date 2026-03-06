"""Core training flow."""

from __future__ import annotations
import time
from typing import Dict
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from vega_utils.anchor_utils import get_anchors
from vega_utils.checkpoint_utils import save_best_checkpoint
from model import MaskedCELoss, MaskedKLDivLoss, Transformer_Based_Model
from vega_utils.report_utils import classification_report_to_df
from train import (
    get_IEMOCAP_loaders,
    get_MELD_loaders,
    print_best_metric,
    print_metrics,
    train_or_eval_model,
)
from vega_utils.common import emotion_labels


def create_model(args) -> torch.nn.Module:
    """Build model and move to GPU if available."""
    model = Transformer_Based_Model(
        args,
        args.Dataset,
        args.text_dim,
        args.visual_dim,
        args.audio_dim,
        args.n_head,
        n_classes=args.n_classes,
        hidden_dim=args.hidden_dim,
        n_speakers=args.n_speakers,
        dropout=args.dropout,
    )
    if args.cuda:
        model.cuda()
    for param in model.parameters():
        param.requires_grad = True
    return model


def setup_optimizer_and_scheduler(args, model: torch.nn.Module):
    """Create optimizer and optional cosine scheduler."""
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = None
    if args.scheduler:
        num_training_steps = args.epochs * args.train_loader_len
        num_warmup_steps = int(0.01 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    return optimizer, scheduler


def setup_data_and_loss(args):
    """Create dataloaders and loss functions."""
    if args.Dataset == "MELD":
        train_loader, test_loader = get_MELD_loaders(
            args,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        loss_weights = torch.Tensor([1.0, 3.93398523, 17.97765350, 6.42315388, 2.78856158, 17.82825470, 4.00497818])
    elif args.Dataset == "IEMOCAP":
        train_loader, test_loader = get_IEMOCAP_loaders(
            args,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        loss_weights = torch.Tensor([2.85339499, 1.70571959, 1.08255267, 1.67633724, 1.77617681, 1.0])
    else:
        raise ValueError(f"Unknown dataset: {args.Dataset}")

    if args.cuda:
        loss_weights = loss_weights.cuda()

    loss_function = MaskedCELoss(loss_weights)
    kl_loss = MaskedKLDivLoss()
    return train_loader, test_loader, loss_function, kl_loss


def train(
    args,
    model: torch.nn.Module,
    anchor_dict: Dict,
    loss_function,
    kl_loss,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
):
    """Run training/evaluation loop and track best checkpoint by cls F1."""
    best_metrics = {
        "best_cls_f1": None,
        "best_cls_pred": None,
        "best_cls_df": None,
    }
    metrics_history = {
        "all_acc_list": [],
        "all_f1_list": [],
        "a_f1_list": [],
        "v_f1_list": [],
        "t_f1_list": [],
    }
    best_ckp = {"cls_f1": None}
    stop_count = 0
    expr_labels = emotion_labels[args.Dataset]

    for epoch_idx in range(args.epochs):
        improved = False
        start_time = time.time()

        train_results = train_or_eval_model(
            args=args,
            model=model,
            anchor_dict=anchor_dict,
            loss_function=loss_function,
            kl_loss_fn=kl_loss,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            is_train=True,
        )
        test_results = train_or_eval_model(
            args=args,
            model=model,
            anchor_dict=anchor_dict,
            loss_function=loss_function,
            kl_loss_fn=kl_loss,
            dataloader=test_loader,
            epoch=epoch_idx,
        )

        test_labels = test_results["labels"]
        test_all_preds = test_results["all_preds"]
        test_masks = test_results["masks"]
        test_all_acc = test_results["all_acc"]
        test_all_f1 = test_results["all_f1"]
        test_a_f1 = test_results["a_f1"]
        test_v_f1 = test_results["v_f1"]
        test_t_f1 = test_results["t_f1"]

        metrics_history["all_acc_list"].append(test_all_acc)
        metrics_history["all_f1_list"].append(test_all_f1)
        metrics_history["a_f1_list"].append(test_a_f1)
        metrics_history["v_f1_list"].append(test_v_f1)
        metrics_history["t_f1_list"].append(test_t_f1)

        if test_all_f1 is not None:
            if best_metrics["best_cls_f1"] is None or best_metrics["best_cls_f1"] < test_all_f1:
                improved = True
                stop_count = 0
                best_metrics["best_cls_f1"] = test_all_f1
                best_metrics["best_cls_pred"] = test_all_preds
                best_metrics["best_cls_df"] = classification_report_to_df(
                    test_labels,
                    test_all_preds,
                    test_masks,
                    expr_labels,
                )
                best_cls_ckp = save_best_checkpoint(
                    args=args,
                    epoch=epoch_idx + 1,
                    model=model,
                    value=best_metrics["best_cls_f1"],
                    metric="cls_f1",
                )
                if best_cls_ckp is not None:
                    best_ckp["cls_f1"] = best_cls_ckp

        print_metrics("Train", epoch_idx + 1, start_time, train_results)
        print_metrics("Test", epoch_idx + 1, start_time, test_results, elapsed_time=True)

        print("\n========================== Best Test performance ==========================")
        for name, values in metrics_history.items():
            if name.endswith("_list") and values[0] is not None:
                print_best_metric(name[:-5], values)

        print()
        print(best_metrics["best_cls_df"])
        print()

        if not improved:
            stop_count += 1

        print("==============================Early Stop==============================")
        print("Current Epoch: ", epoch_idx, " STOP COUNT:", stop_count)

        if stop_count > 30:
            break

    return best_metrics


def main(args):
    """Main training flow."""
    anchor_dict = get_anchors(args) if args.clip_loss else None
    train_loader, test_loader, loss_function, kl_loss = setup_data_and_loss(args)
    args.train_loader_len = len(train_loader)

    model = create_model(args)
    optimizer, scheduler = setup_optimizer_and_scheduler(args, model)
    best_metrics = train(
        args=args,
        model=model,
        anchor_dict=anchor_dict,
        loss_function=loss_function,
        kl_loss=kl_loss,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    print("\n===== Training Complete =====")
    print(f"Best CLS F1: {best_metrics['best_cls_f1']:.4f}")
    return best_metrics["best_cls_f1"]
