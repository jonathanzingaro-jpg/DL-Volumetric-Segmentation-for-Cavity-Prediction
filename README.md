# Deep Learning Volumetric Segmentation for Cavity Prediction

Code for the paper:

> G. Zingaro, Y. Wang, D. Meena, P. Cheng, R. Buchanan, J. Quilty, R. Gracie (2026). **Deep Learning Volumetric Segmentation for Prediction of Cavity Geometry in Underground Jet Boring Mining.** arXiv:2610.36303.

## Overview

The Jet Boring System (JBS), developed by Cameco Corporation for the Cigar Lake uranium mine in Saskatchewan, Canada, excavates ore using a high-pressure water jet that cuts cavities within an artificially frozen ore body. This repository predicts the resulting cavity geometry with deep learning.

- **Preprocessing:** drilling and jetting records, cavity scans, and the mine-wide geology block model are assembled onto a common voxel grid for each cavity.
- **Model:** a 3D U-Net with a backfill residual branch classifies each voxel as consumed or non-consumed during jetting.
- **Baseline:** a cylindrical cavity centred on the pilot hole, with a diameter set by domain experts.
- **Explainability:** XAI feature attribution identifies which inputs most influence the predicted cavity geometry.

On a held-out test set, the model achieves an IoU of 0.601 ± 0.088 and an F₁-score of 0.747 ± 0.077, improving on the cylindrical baseline by 22.4% and 14.7%, respectively.

## Repository structure

```
├── Config/          # Model and training configuration
├── Functions/       # Preprocessing, model, training and evaluation functions
├── Run Files/       # Scripts to train and evaluate the models
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The field data used in this study (drilling and jetting records, cavity scans, and the geology block model) are proprietary to Cameco Corporation and are not included in this repository.

## Citation

```bibtex
@article{zingaro2026cavity,
  title   = {Deep Learning Volumetric Segmentation for Prediction of Cavity Geometry in Underground Jet Boring Mining},
  author  = {Zingaro, Giovanni and Wang, Yuan and Meena, Deepak and Cheng, Peter and Buchanan, Russell and Quilty, John and Gracie, Robert},
  journal = {arXiv preprint arXiv:2610.36303},
  year    = {2026}
}
```
