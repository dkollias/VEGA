import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle


def _dataset_path(dataset_name):
    if dataset_name == 'IEMOCAP':
        dataset = 'data/IEMOCAP.pkl'
    else:
        dataset = 'data/meld.pkl'

    return dataset


def _load_dataset_payload(path):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return payload


def _feature_dim_mask(tensor, ratio):
    if ratio <= 0:
        return tensor
    if tensor.dim() == 1:
        feat_dim = tensor.shape[0]
        if feat_dim == 0:
            return tensor
        mask = torch.rand(feat_dim, device=tensor.device) >= ratio
        return tensor * mask
    feat_dim = tensor.shape[-1]
    if feat_dim == 0:
        return tensor
    mask = (torch.rand(feat_dim, device=tensor.device) >= ratio).to(tensor.dtype)
    return tensor * mask


class IEMOCAPDataset(Dataset):
    def __init__(self, args, train=True):
        path = _dataset_path('IEMOCAP')
        (
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.trainVid,
            self.testVid,
        ) = _load_dataset_payload(path)

        self._apply_mask = train
        self._mask_ratio = getattr(args, 'aug_feature_mask_ratio', 0.1)

        self.keys = list(self.trainVid if train else self.testVid)
        self._speaker_cache = {}
        self._mask_cache = {}
        for vid in self.keys:
            speaker = np.asarray(
                [[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]],
                dtype=np.float32
            )
            self._speaker_cache[vid] = torch.from_numpy(speaker)
            self._mask_cache[vid] = torch.ones(len(self.videoLabels[vid]), dtype=torch.float32)

    def __getitem__(self, index):
        vid = self.keys[index]
        text = self._to_float_tensor(self.videoText[vid]).squeeze(0)
        visual = self._to_float_tensor(self.videoVisual[vid])
        audio = self._to_float_tensor(self.videoAudio[vid])
        if self._apply_mask and self._mask_ratio > 0:
            text = _feature_dim_mask(text, self._mask_ratio)
            visual = _feature_dim_mask(visual, self._mask_ratio)
            audio = _feature_dim_mask(audio, self._mask_ratio)
        speakers = self._speaker_cache[vid]
        umask = self._mask_cache[vid]
        labels = torch.as_tensor(self.videoLabels[vid], dtype=torch.long)
        return text, \
            visual, \
            audio, \
            speakers, \
            umask, \
            labels, \
            vid

    def __len__(self):
        return len(self.keys)

    def collate_fn(self, data):
        text, visual, audio, speakers, umask, labels, vids = zip(*data)
        return [
            pad_sequence(text),
            pad_sequence(visual),
            pad_sequence(audio),
            pad_sequence(speakers),
            pad_sequence(umask, True, 0),
            pad_sequence(labels, True, -1),
            list(vids),
        ]

    @staticmethod
    def _to_float_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).float()
        return torch.from_numpy(np.asarray(value)).float()


class MELDDataset(Dataset):
    def __init__(self, args, train=True):
        path = _dataset_path('MELD')
        (
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.trainVid,
            self.testVid,
        ) = _load_dataset_payload(path)

        self._apply_mask = train
        self._mask_ratio = getattr(args, 'aug_feature_mask_ratio', 0.1)

        self.keys = list(self.trainVid if train else self.testVid)
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        text = self._to_float_tensor(self.videoText[vid])
        visual = self._to_float_tensor(self.videoVisual[vid])
        audio = self._to_float_tensor(self.videoAudio[vid])
        if self._apply_mask and self._mask_ratio > 0:
            text = _feature_dim_mask(text, self._mask_ratio)
            visual = _feature_dim_mask(visual, self._mask_ratio)
            audio = _feature_dim_mask(audio, self._mask_ratio)
        speakers = self._to_float_tensor(self.videoSpeakers[vid])
        return text, \
            visual, \
            audio, \
            speakers, \
            torch.FloatTensor([1] * len(self.videoLabels[vid])), \
            torch.LongTensor(self.videoLabels[vid]), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        import pandas as pd
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i < 6:
                if i < 4:
                    return_list.append(pad_sequence(dat[i]))
                elif i == 4:
                    return_list.append(pad_sequence(dat[i], True, 0))
                elif i == 5:
                    return_list.append(pad_sequence(dat[i], True, -1))
            else:
                return_list.append(dat[i].tolist())

        return return_list

    @staticmethod
    def _to_float_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).float()
        return torch.from_numpy(np.asarray(value)).float()
