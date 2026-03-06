# output/

This folder stores training outputs generated at runtime.

## What goes here

- Experiment logs
- Metrics reports
- Intermediate/final outputs created during training or evaluation

## Usage

Run training from the project root, and outputs will be written here as configured.

Example:

```bash
python run.py --Dataset IEMOCAP --CLIP_Model openai/clip-vit-base-patch32 --cls_loss --clip_loss --clip_all_clip_kl_loss --cls_all_cls_kl_loss
```

## Note

- This repository tracks this folder and this README for structure.
- Large generated files remain ignored by `.gitignore` by default.
