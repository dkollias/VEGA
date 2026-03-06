# VEGA: Grounding Emotion Recognition with Visual Prototypes


This repository provides the **official implementation** of our ACM MM 2025 paper:

> **[Grounding Emotion Recognition with Visual Prototypes: VEGA--Revisiting CLIP in MERC](#)**  
> **Guanyu Hu, Dimitrios Kollias, Xinyu Yang**


## Overview ✨

VEGA addresses Multimodal Emotion Recognition in Conversations (MERC) with explicit visual semantic grounding.
Instead of only feature fusion, VEGA introduces CLIP-based visual emotion anchors and aligns unimodal/fused representations to anchor prototypes.

**Core ideas**:

1. Build class-level visual prototypes in CLIP space.
2. Align text/audio/visual/fused representations to prototypes.
3. Optimize with classification + distillation in both label and anchor spaces.

## Model Overview 🖼️

![VEGA Model](Model.png)

## Method at a Glance 🧠

VEGA is trained with two collaborative branches:

### 1. **Supervision Branch**
- Classification losses for unimodal and fused outputs.
- Distillation from fused prediction to unimodal predictions.

### 2. **VEGA Branch**
- Projects multimodal utterance representations into a visual semantic space.
- Performs **visual semantic anchoring** by aligning utterance features to visual anchors.
- Optimizes anchor-conditioned classification and semantic distillation to enforce anchor-grounded decision boundaries.

## Repository Layout 🗂️

```text
VEGA/
├─ run.py                 # Training entrypoint
├─ main.py                # Main training/evaluation flow
├─ train.py               # Optimization and metrics
├─ model.py               # Backbone + VEGA modules
├─ inference.py           # Inference script
├─ dataloader.py          # Runtime dataloader
├─ configs/               # Configuration presets
├─ vega_utils/            # Utilities (anchors/checkpoints/reports/common)
├─ docs/figures/          # Figures used in README
├─ data/                  # Multimodal feature files
├─ anchor/                # Visual anchor assets/cache
├─ checkpoint/            # Pretrained checkpoints
└─ output/                # Training outputs
```

## Environment ⚙️

- Python `3.10+`
- Core packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `transformers`, `pillow`, `pytz`

```bash
pip install torch numpy pandas scikit-learn tqdm transformers pillow pytz
```

## Data Preparation ⬇️

Place files exactly as follows:

| Resource                                     | Filename | Target Path | Google Drive |
|----------------------------------------------|---|---|---|
| Visual anchors (anchor images, optional)     | `35_anchor.zip` | unzip to `anchor/35_anchor/` | `https://drive.google.com/file/d/1DOmYn6tISoEPJ4PQDD4F-gB1M58G-NS1/view?usp=sharing` |
| Visual anchors (ready features, recommended) | `35_anchor.pt` | `anchor/35_anchor.pt` | `https://drive.google.com/file/d/1F-ajsUUHihO0RgREros5AJUiptuVOoIl/view?usp=sharing` |
| Multimodal features                          | `IEMOCAP.pkl` | `data/IEMOCAP.pkl` | `https://drive.google.com/file/d/1dx4yikoU8hYZ7FxyrRwzcANzaJBcg90N/view?usp=sharing` |
| Checkpoint                                   | `IEMOCAP.pth` | `checkpoint/IEMOCAP.pth` | `https://drive.google.com/file/d/1piNYmb1GfRNruKkDHf6_GSUh1ctEYvul/view?usp=sharing` |

If you choose `35_anchor.zip`, extract/build the anchor cache and save it to `anchor/35_anchor.pt`.
When building from raw images on IEMOCAP, ensure the folder includes images for all labels:
`happy`, `sad`, `neutral`, `anger`, `excited`, `frustration`.

<details>
<summary><b>Required structure of <code>35_anchor.pt</code></b></summary>

```python
{
    "anchor_center": torch.Tensor,  # [num_classes, clip_dim], e.g. [6, 512]
    "anchor_img_dict": {
        "happy": {"feature": torch.Tensor},        # [num_images, clip_dim]
        "sad": {"feature": torch.Tensor},
        "neutral": {"feature": torch.Tensor},
        "anger": {"feature": torch.Tensor},
        "excited": {"feature": torch.Tensor},
        "frustration": {"feature": torch.Tensor},
    },
}
```

Notes:
- `feature` and `anchor_center` are CLIP visual features.
- CPU/GPU tensors are both acceptable when saving.

</details>

## Quick Start 🚀

### Training 🏋️

```bash
python run.py --Dataset IEMOCAP --CLIP_Model openai/clip-vit-base-patch32 --cls_loss --clip_loss --clip_all_clip_kl_loss --cls_all_cls_kl_loss
```

### Inference 🔍

```bash
python inference.py --checkpoint "checkpoint/IEMOCAP.pth"
```

## Citation 📚

```bibtex
@inproceedings{hu2025grounding,
  title={Grounding Emotion Recognition with Visual Prototypes: VEGA - Revisiting CLIP in MERC},
  author={Hu, Guanyu and Kollias, Dimitrios and Yang, Xinyu},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={5667--5676},
  year={2025},
  doi={10.1145/3746027.3755340}
}
```

## Acknowledgements 

This project builds on excellent open-source foundations.  

- SDT: https://github.com/butterfliesss/SDT
- CLIP: https://github.com/openai/CLIP
