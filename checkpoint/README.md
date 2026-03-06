# checkpoint/

This folder stores pretrained or trained model checkpoints.

## Official checkpoint

- Download `IEMOCAP.pth` from the Google Drive link listed in the root [`README.md`](../README.md).
- Place it at:

```text
checkpoint/IEMOCAP.pth
```

## Inference usage

Example:

```bash
python inference.py --checkpoint "checkpoint/IEMOCAP.pth"
```
