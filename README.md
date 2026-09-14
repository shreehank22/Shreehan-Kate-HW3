# RBE 4540/595 - Homework 3: Grasp Quality Analysis

Planar grasp matrix derivation and grasp quality metric comparison for a
4-contact hard-finger grasp on a 6cm x 3cm rectangular object, sweeping
a 5th contact location around the boundary.

## Contents
- `grasp_hw3.py` — grasp matrix function, boundary sweep, quality metrics, plots
- `results/grasp_quality_vs_position.png` — metric value vs. 5th-contact arc-length position
- `results/best_locations.png` — optimal 5th-contact location per metric, drawn on the object
- `report/report.pdf` — full writeup: hand derivation, plots, discussion

## Metrics implemented (Roa & Suarez, "Grasp Quality Measures: Review and Performance")
- 3.1.1 — minimum singular value of G_p
- 3.1.2 — volume of the wrench-space ellipsoid (product of singular values)
- 3.1.3 — grasp isotropy index (sigma_min / sigma_max)

## Run
```
pip install -r requirements.txt
python grasp_hw3.py
```

## Note
Bottom-contact x-position (`p4x` in the script) was assumed centered on the
bottom edge based on the assignment figure — verify against the actual
figure before final submission.
