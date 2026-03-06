import random
from argparse import Namespace
import torch.nn as nn
import math
from functools import partial
from vega_utils.common import emotion_labels


def build_proj(input_dim, output_dim, dropout, layer_num=1, hidden_dim=None,
               activation_fn='silu', layer_type='conv1d'):
    activation_map = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'none': nn.Identity
    }

    layer_map = {
        'linear': nn.Linear,
        'conv1d': nn.Conv1d,
    }

    if activation_fn is not None:
        act_layer = activation_map[activation_fn.lower()]
    layer_cls = layer_map[layer_type.lower()]

    if hidden_dim is None:
        hidden_dim = input_dim

    layers = []

    for i in range(layer_num):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == layer_num - 1 else hidden_dim
        is_last_layer = (i == layer_num - 1)

        if layer_type == 'linear':
            layer = layer_cls(in_dim, out_dim)
        elif layer_type == 'conv1d':
            layer = layer_cls(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        layers.append(layer)

        if not is_last_layer and activation_fn is not None:
            layers.append(act_layer())

        if dropout > 0 and not is_last_layer:
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(self, student_logits, teacher_logits, temp, mask):
        mask_ = mask.view(-1, 1).expand_as(teacher_logits).to(torch.int).bool()

        teacher_probs = F.softmax(teacher_logits / temp, dim=1)
        student_log_probs = F.log_softmax(student_logits / temp, dim=1)

        kl_loss = self.kl_loss(student_log_probs, teacher_probs) * (temp ** 2)

        return (kl_loss[mask_]).mean()


class MaskedCELoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedCELoss, self).__init__()
        self.weight = weight
        self.loss = nn.CrossEntropyLoss(weight=weight, reduction='mean', ignore_index=-1)

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1).to(torch.int).bool()
        loss = self.loss(pred[mask_], target[mask_])
        return loss


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None, expr=False):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            if expr:
                mask = mask.unsqueeze(1).unsqueeze(-1).expand_as(scores).to(torch.bool)
                scores = scores.masked_fill(mask, -1e4)
            else:
                mask = mask.unsqueeze(1).expand_as(scores)
                scores = scores.masked_fill(mask, -1e4)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep


class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep


import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert("RGB")
        return img


def collate_fn(batch, clip_processor):
    inputs = clip_processor(images=batch, return_tensors="pt")
    return inputs


def get_clip_visual_features_batch(img_list, clip_model, clip_processor, batch_size=32):
    if not img_list:
        raise ValueError("Empty image list for anchor feature extraction.")

    device = next(clip_model.parameters()).device
    pin_memory = device.type == "cuda"

    dataset = ImageDataset(img_list)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, clip_processor=clip_processor),
        num_workers=0,
        pin_memory=pin_memory,
    )
    all_features = []
    clip_model.eval()
    ctx = torch.inference_mode()
    with ctx:
        for image_input in tqdm(dataloader):
            image_input = {k: v.to(device, non_blocking=pin_memory) for k, v in image_input.items()}
            image_features = clip_model.get_image_features(**image_input)
            all_features.append(F.normalize(image_features, dim=-1))
    return torch.cat(all_features, dim=0)





class Transformer_Based_Model(nn.Module):
    def __init__(self, args: Namespace, dataset, text_dim, visual_dim, audio_dim, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
        super(Transformer_Based_Model, self).__init__()
        self.args = args
        out_layer_dim = hidden_dim

        self.a_cls_temp = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.v_cls_temp = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.t_cls_temp = nn.Parameter(torch.log(torch.tensor(2.0)))

        self.a_clip_temp = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.v_clip_temp = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.t_clip_temp = nn.Parameter(torch.log(torch.tensor(2.0)))

        self.n_classes = n_classes
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)

        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        self.textf_input = nn.Conv1d(text_dim, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(audio_dim, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(visual_dim, hidden_dim, kernel_size=1, padding=0, bias=False)

        if args.clip_loss:
            self.all_clip_proj = build_proj(hidden_dim, args.clip_dim, args.clip_proj_drop,
                                            layer_num=args.clip_proj_layer_num,
                                            activation_fn=args.clip_proj_activation_fn,
                                            layer_type='linear')
            self.t_clip_proj = build_proj(hidden_dim, args.clip_dim, args.clip_proj_drop,
                                          layer_num=args.clip_proj_layer_num,
                                          activation_fn=args.clip_proj_activation_fn, layer_type='linear')
            self.a_clip_proj = build_proj(hidden_dim, args.clip_dim, args.clip_proj_drop,
                                          layer_num=args.clip_proj_layer_num,
                                          activation_fn=args.clip_proj_activation_fn, layer_type='linear')
            self.v_clip_proj = build_proj(hidden_dim, args.clip_dim, args.clip_proj_drop,
                                          layer_num=args.clip_proj_layer_num,
                                          activation_fn=args.clip_proj_activation_fn, layer_type='linear')

        self.t_output_layer = build_proj(out_layer_dim, n_classes, args.outlayer_drop,
                                         layer_num=args.outlayer_num,
                                         activation_fn=args.outlayer_activation_fn,
                                         layer_type='linear')
        self.a_output_layer = build_proj(out_layer_dim, n_classes, args.outlayer_drop,
                                         layer_num=args.outlayer_num,
                                         activation_fn=args.outlayer_activation_fn,
                                         layer_type='linear')
        self.v_output_layer = build_proj(out_layer_dim, n_classes, args.outlayer_drop,
                                         layer_num=args.outlayer_num,
                                         activation_fn=args.outlayer_activation_fn,
                                         layer_type='linear')
        self.all_output_layer = build_proj(out_layer_dim, n_classes, args.outlayer_drop,
                                           layer_num=args.outlayer_num,
                                           activation_fn=args.outlayer_activation_fn,
                                           layer_type='linear')

        self.prob_gate_logit = nn.Parameter(torch.tensor(0.0))

    def _build_speaker_embeddings(self, qmask, dia_len):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        return self.speaker_embeddings(spk_idx)

    def _project_modal_inputs(self, textf, visuf, acouf):
        textf = self.textf_input(textf).transpose(1, 2)
        acouf = self.acouf_input(acouf).transpose(1, 2)
        visuf = self.visuf_input(visuf).transpose(1, 2)
        return textf, visuf, acouf

    def _forward_transformer_branch(self, textf, visuf, acouf, u_mask, spk_embeddings):
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)

        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.v_t_gate(v_t_transformer_out)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)

        t_transformer_out = self.features_reduce_t(
            torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(
            torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(
            torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))


        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)
        return t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out

    def _forward_backbone_logits(self, t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out):
        t_logit = self.t_output_layer(t_transformer_out)
        a_logit = self.a_output_layer(a_transformer_out)
        v_logit = self.v_output_layer(v_transformer_out)
        all_logit = self.all_output_layer(all_transformer_out)
        return t_logit, a_logit, v_logit, all_logit

    def _get_temperature_values(self):
        a_cls_temp = self.a_cls_temp.exp()
        v_cls_temp = self.v_cls_temp.exp()
        t_cls_temp = self.t_cls_temp.exp()

        a_clip_temp = self.a_clip_temp.exp()
        v_clip_temp = self.v_clip_temp.exp()
        t_clip_temp = self.t_clip_temp.exp()
        return a_cls_temp, v_cls_temp, t_cls_temp, a_clip_temp, v_clip_temp, t_clip_temp

    def _select_expr_feature(self, anchor_img_dict, anchor_center, train):
        if train:
            i = random.randint(1, 100)
            if i <= self.args.rand * 100:
                expr_feature = []
                for expr in emotion_labels[self.args.Dataset]:
                    feature_bank = anchor_img_dict[expr]['feature']
                    img_num = feature_bank.size(0) if isinstance(feature_bank, torch.Tensor) else len(feature_bank)
                    rand_n = random.randint(0, img_num - 1)
                    expr_feature.append(feature_bank[rand_n].unsqueeze(0))
                expr_feature = torch.cat(expr_feature)
            else:
                expr_feature = anchor_center
        else:
            expr_feature = anchor_center
        expr_feature = F.normalize(expr_feature, dim=-1)
        return expr_feature.unsqueeze(0).unsqueeze(0)

    def _forward_vega_logits(self, anchor_img_dict, anchor_center,
                             t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out, train):
        expr_feature = self._select_expr_feature(anchor_img_dict, anchor_center, train)

        a_clip_proj_out = self.a_clip_proj(a_transformer_out)
        norm_a_proj_feature = F.normalize(a_clip_proj_out, dim=-1).unsqueeze(2)

        v_clip_proj_out = self.v_clip_proj(v_transformer_out)
        norm_v_proj_feature = F.normalize(v_clip_proj_out, dim=-1).unsqueeze(2)

        t_clip_proj_out = self.t_clip_proj(t_transformer_out)
        norm_t_proj_feature = F.normalize(t_clip_proj_out, dim=-1).unsqueeze(2)

        all_clip_proj_out = self.all_clip_proj(all_transformer_out)
        norm_all_proj_feature = F.normalize(all_clip_proj_out, dim=-1).unsqueeze(2)

        a_clip_logit = (norm_a_proj_feature * expr_feature).sum(-1) * 100
        v_clip_logit = (norm_v_proj_feature * expr_feature).sum(-1) * 100
        t_clip_logit = (norm_t_proj_feature * expr_feature).sum(-1) * 100
        all_clip_logit = (norm_all_proj_feature * expr_feature).sum(-1) * 100
        all_clip_prob = F.softmax(all_clip_logit, -1)
        return t_clip_logit, a_clip_logit, v_clip_logit, all_clip_logit, all_clip_prob

    def _fuse_backbone_clip_prob(self, all_prob, all_clip_prob):
        gate = torch.sigmoid(self.prob_gate_logit)
        average_prob = gate * all_prob + (1 - gate) * all_clip_prob
        max_prob = torch.where(
            all_prob.max(dim=-1, keepdim=True).values > all_clip_prob.max(dim=-1, keepdim=True).values,
            all_prob,
            all_clip_prob,
        )
        return average_prob, max_prob

    def forward_backbone(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        spk_embeddings = self._build_speaker_embeddings(qmask, dia_len)
        textf, visuf, acouf = self._project_modal_inputs(textf, visuf, acouf)
        t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out = self._forward_transformer_branch(
            textf, visuf, acouf, u_mask, spk_embeddings
        )
        t_logit, a_logit, v_logit, all_logit = self._forward_backbone_logits(
            t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out
        )
        return t_logit, a_logit, v_logit, all_logit, all_transformer_out

    def forward(self, anchor_dict, textf, visuf, acouf, u_mask, qmask, dia_len, train):
        anchor_img_dict, anchor_center = None, None
        if anchor_dict is not None:
            anchor_img_dict = anchor_dict.get('anchor_img_dict')
            anchor_center = anchor_dict.get('anchor_center')
        elif self.args.clip_loss:
            raise ValueError("anchor_dict is required when clip_loss is enabled.")

        average_prob, max_prob = None, None
        fusion_logit, fusion_prob = None, None

        t_clip_logit, a_clip_logit, v_clip_logit, all_clip_logit = None, None, None, None

        (
            a_cls_temp, v_cls_temp, t_cls_temp,
            a_clip_temp, v_clip_temp, t_clip_temp
        ) = self._get_temperature_values()

        all_clip_prob = None
        if self.args.clip_loss:
            if anchor_img_dict is None or anchor_center is None:
                raise ValueError("anchor_dict with 'anchor_img_dict' and 'anchor_center' is required for clip_loss.")
            spk_embeddings = self._build_speaker_embeddings(qmask, dia_len)
            textf, visuf, acouf = self._project_modal_inputs(textf, visuf, acouf)
            t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out = self._forward_transformer_branch(
                textf, visuf, acouf, u_mask, spk_embeddings
            )
            (
                t_clip_logit, a_clip_logit, v_clip_logit, all_clip_logit, all_clip_prob
            ) = self._forward_vega_logits(
                anchor_img_dict, anchor_center, t_transformer_out, a_transformer_out, v_transformer_out,
                all_transformer_out, train
            )
            t_logit, a_logit, v_logit, all_logit = self._forward_backbone_logits(
                t_transformer_out, a_transformer_out, v_transformer_out, all_transformer_out
            )
        else:
            t_logit, a_logit, v_logit, all_logit, all_transformer_out = self.forward_backbone(
                textf, visuf, acouf, u_mask, qmask, dia_len
            )

        all_prob = F.softmax(all_logit, -1)

        if self.args.clip_loss:
            average_prob, max_prob = self._fuse_backbone_clip_prob(all_prob, all_clip_prob)

        return (
            t_logit, a_logit, v_logit, all_logit,
            a_cls_temp, v_cls_temp, t_cls_temp,

            t_clip_logit, a_clip_logit, v_clip_logit, all_clip_logit,
            a_clip_temp, v_clip_temp, t_clip_temp,

            fusion_logit, fusion_prob, average_prob, max_prob,
            all_transformer_out
        )
