import torch
import numpy as np, time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from collections import Counter

from dataloader import IEMOCAPDataset, MELDDataset
from sklearn.metrics import f1_score, precision_score


def compute_class_weights_from_labels(label_list, num_classes):
    counts = Counter(l for l in label_list if l >= 0)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        weights.append(1.0 / count if count > 0 else 0.0)

    non_zero = [w for w in weights if w > 0]
    if non_zero:
        min_w = min(non_zero)
        weights = [w / min_w if w > 0 else 0.0 for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def print_best_metric(metric_name, metric_list):
    best_value = max(metric_list)
    best_epoch = metric_list.index(best_value) + 1
    print(f'{metric_name}: {best_value:.2f}, idx: {best_epoch}')


def print_metrics(phase, epoch, start_time, metrics, elapsed_time=False):
    print(f'\n===================== {phase} ========================')
    print(
        f"epoch: {epoch + 1}, loss: {metrics['loss']}, cls_loss: {metrics['cls_loss']}, "
        f"cls_kl_loss: {metrics['cls_kl_loss']}"
    )
    if metrics.get('clip_loss') is not None:
        print(
            f"clip_loss: {metrics['clip_loss']}, "
            f"clip_kl_loss: {metrics['clip_kl_loss']}"
        )
    print(
        f"all_acc: {metrics['all_acc']}, all_f1: {metrics['all_f1']}, a_f1: {metrics['a_f1']}, "
        f"v_f1: {metrics['v_f1']}, t_f1: {metrics['t_f1']}"
    )
    if elapsed_time:
        print(f'Time: {round(time.time() - start_time, 2)}s')


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(args, batch_size=32, num_workers=0, pin_memory=False):
    trainset = MELDDataset(args)
    train_sampler, _ = get_train_valid_sampler(trainset, 0.0, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = MELDDataset(args, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader


def get_IEMOCAP_loaders(args, batch_size=32, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(args)
    train_sampler, _ = get_train_valid_sampler(trainset, 0.0)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    labels = [l for labels in trainset.videoLabels.values() for l in labels]
    weights = compute_class_weights_from_labels(labels, num_classes=6)
    print(weights)

    testset = IEMOCAPDataset(args, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader


def train_or_eval_model(args, model, anchor_dict, loss_function, kl_loss_fn, dataloader, optimizer=None, scheduler=None,
                        is_train=False, epoch=None):
    all_preds, labels, masks = [], [], []
    a_preds, v_preds, t_preds = [], [], []
    losses, cls_losses, cls_all_cls_kl_losses = [], [], []
    clip_losses, clip_all_clip_kl_losses = [], []

    scaler = torch.amp.GradScaler('cuda', enabled=args.cuda)
    assert (not is_train) or (optimizer is not None)
    model.train() if is_train else model.eval()

    for data in tqdm(dataloader):
        if is_train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        vid_list = data[-1]

        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        textf = textf.permute(1, 2, 0)
        acouf = acouf.permute(1, 2, 0)
        visuf = visuf.permute(1, 2, 0)

        with torch.amp.autocast("cuda", enabled=args.cuda):
            forward_train_flag = is_train or getattr(args, 'eval_forward_train_flag', True)
            (
                t_logit, a_logit, v_logit, all_logit,
                a_cls_temp, v_cls_temp, t_cls_temp,
                t_clip_logit, a_clip_logit, v_clip_logit, all_clip_logit,
                a_clip_temp, v_clip_temp, t_clip_temp,
                *_
            ) = model(anchor_dict, textf, visuf, acouf, umask, qmask, lengths, forward_train_flag)

            labels_ = label.view(-1)
            max_len = label.size(1)
            t_logit = t_logit.view(-1, t_logit.size(2))
            a_logit = a_logit.view(-1, a_logit.size(2))
            v_logit = v_logit.view(-1, v_logit.size(2))
            all_logit = all_logit.view(-1, all_logit.size(2))

            all_loss = loss_function(all_logit, labels_, umask) * args.cls_all_lambda
            t_loss = loss_function(t_logit, labels_, umask) * args.cls_t_lambda
            a_loss = loss_function(a_logit, labels_, umask) * args.cls_a_lambda
            v_loss = loss_function(v_logit, labels_, umask) * args.cls_v_lambda
            cls_loss = all_loss + a_loss + t_loss + v_loss
            loss = cls_loss * args.cls_lambda
            cls_losses.append(cls_loss.item())

            cls_kl_loss = 0
            if args.cls_all_cls_kl_loss:
                t_kl_all_cls_loss = kl_loss_fn(t_logit, all_logit, t_cls_temp, umask) * args.t_cls_all_cls_kl_lambda
                a_kl_all_cls_loss = kl_loss_fn(a_logit, all_logit, a_cls_temp, umask) * args.a_cls_all_cls_kl_lambda
                v_kl_all_cls_loss = kl_loss_fn(v_logit, all_logit, v_cls_temp, umask) * args.v_cls_all_cls_kl_lambda
                cls_kl_loss = t_kl_all_cls_loss + a_kl_all_cls_loss + v_kl_all_cls_loss
                loss += cls_kl_loss * args.cls_all_cls_kl_lambda
                cls_all_cls_kl_losses.append(cls_kl_loss.item())

            if args.clip_loss:
                t_clip_logit = t_clip_logit.view(-1, t_clip_logit.size(2))
                a_clip_logit = a_clip_logit.view(-1, a_clip_logit.size(2))
                v_clip_logit = v_clip_logit.view(-1, v_clip_logit.size(2))
                all_clip_logit = all_clip_logit.view(-1, all_clip_logit.size(2))

                t_clip_loss = loss_function(t_clip_logit, labels_, umask) * args.t_clip_lambda
                a_clip_loss = loss_function(a_clip_logit, labels_, umask) * args.a_clip_lambda
                v_clip_loss = loss_function(v_clip_logit, labels_, umask) * args.v_clip_lambda
                all_clip_loss = loss_function(all_clip_logit, labels_, umask) * args.all_clip_lambda
                clip_loss = a_clip_loss + t_clip_loss + v_clip_loss + all_clip_loss
                loss += clip_loss * args.clip_lambda
                clip_losses.append(clip_loss.item())

                if args.clip_all_clip_kl_loss:
                    t_clip_all_clip_kl_loss = kl_loss_fn(
                        t_clip_logit, all_clip_logit, t_clip_temp, umask
                    ) * args.t_clip_all_clip_kl_lambda
                    a_clip_all_clip_kl_loss = kl_loss_fn(
                        a_clip_logit, all_clip_logit, a_clip_temp, umask
                    ) * args.a_clip_all_clip_kl_lambda
                    v_clip_all_clip_kl_loss = kl_loss_fn(
                        v_clip_logit, all_clip_logit, v_clip_temp, umask
                    ) * args.v_clip_all_clip_kl_lambda
                    clip_all_clip_kl_loss = (
                        t_clip_all_clip_kl_loss + a_clip_all_clip_kl_loss + v_clip_all_clip_kl_loss
                    )
                    loss += clip_all_clip_kl_loss * args.clip_all_clip_kl_lambda
                    clip_all_clip_kl_losses.append(clip_all_clip_kl_loss.item())

            losses.append(loss.item())

        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        all_preds.append(torch.argmax(all_logit, 1).data.cpu().numpy())
        a_preds.append(torch.argmax(a_logit, 1).data.cpu().numpy())
        v_preds.append(torch.argmax(v_logit, 1).data.cpu().numpy())
        t_preds.append(torch.argmax(t_logit, 1).data.cpu().numpy())

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler:
                scheduler.step()

    all_preds = np.concatenate(all_preds)
    a_preds = np.concatenate(a_preds)
    v_preds = np.concatenate(v_preds)
    t_preds = np.concatenate(t_preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    mask = labels != -1
    epoch_loss = round(sum(losses) / np.sum(masks), 4)
    epoch_cls_loss = round(np.sum(cls_losses) / np.sum(masks), 4)
    epoch_cls_all_cls_kl_loss = round(np.sum(cls_all_cls_kl_losses) / np.sum(masks), 4)
    if args.clip_loss:
        epoch_clip_loss = round(np.sum(clip_losses) / np.sum(masks), 4)
        epoch_clip_all_clip_kl_loss = round(np.sum(clip_all_clip_kl_losses) / np.sum(masks), 4)
    else:
        epoch_clip_loss = None
        epoch_clip_all_clip_kl_loss = None

    all_f1 = round(f1_score(labels[mask], all_preds[mask], average='weighted') * 100, 2)
    all_acc = round(float(precision_score(labels[mask], all_preds[mask], average="weighted", zero_division=0) * 100), 2)

    a_f1 = round(f1_score(labels[mask], a_preds[mask], average='weighted') * 100, 2)
    t_f1 = round(f1_score(labels[mask], t_preds[mask], average='weighted') * 100, 2)
    v_f1 = round(f1_score(labels[mask], v_preds[mask], average='weighted') * 100, 2)

    return {
        'labels': labels,
        'all_preds': all_preds,
        'masks': masks,
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cls_kl_loss': epoch_cls_all_cls_kl_loss,
        'clip_loss': epoch_clip_loss,
        'clip_kl_loss': epoch_clip_all_clip_kl_loss,
        'all_acc': all_acc,
        'all_f1': all_f1,
        'a_f1': a_f1,
        'v_f1': v_f1,
        't_f1': t_f1,
    }
