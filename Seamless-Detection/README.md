# Seamless-Detection

Unified SOD/COD segmentation framework with contrastive distillation.

## Highlights

- Unified learning for Salient Object Detection (SOD) and Camouflaged Object Detection (COD).
- Shared encoder-decoder training with contrastive foreground/background semantics.
- Supports training, testing, CRF post-processing, and metric evaluation.

## Project Layout

```text
Seamless-Detection/
	base/                  # shared framework, encoder backbones, loss factory
	methods/cornet_compare/# method-specific config/model/loss/saver
	dataset/               # put datasets here (see dataset/README.md)
	train.py               # training entry point
	test.py                # testing/inference entry point
	eval.py                # metric evaluation from saved predictions
	crf.py                 # CRF post-processing
	deploy.py              # single-image/folder inference utility
```

## Environment

Recommended:

- Python 3.8-3.10
- CUDA-enabled PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Default expected structure (relative to this repository):

```text
dataset/
	MSB-TR/
		images/
		segmentations/
		found/               # optional pseudo labels
		found_crf/           # optional pseudo labels
	ECSSD/
		images/
		segmentations/
	DUTS-TE/
		images/
		segmentations/
	DUT-OMRON/
		images/
		segmentations/
```

The code reads:

- images from: dataset/<set_name>/images/\*.jpg
- masks from: dataset/<set_name>/<tag>/\*.png

Where <tag> is selected by training/testing flags (segmentations, found, or found_crf).

## Quick Start

### 1) Train

```bash
python train.py cornet_compare \
	--gpus=0 \
	--save \
	--found \
	--vals=ECSSD
```

### 2) Test

```bash
python test.py cornet_compare \
	--weight=./weight/cornet_compare/resnet/base/cornet_compare_base_24.pth \
	--gpus=0 \
	--save \
	--vals=ECSSD,DUTS-TE,DUT-OMRON,PASCAL-S
```

### 3) Evaluate Saved Predictions

```bash
python eval.py --data_path=./dataset --pre_path=./result/cornet_compare/resnet/base --vals=ECSSD,DUTS-TE
```

### 4) Apply CRF Post-processing

```bash
python crf.py MSB-TR
```
---
