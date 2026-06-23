# Guida Utility Detection Pipeline

A computer vision and geospatial pipeline developed during an internship at **Guida Surveying, Inc.** to automate the identification and localization of street-level utilities from imagery. The system detects utilities in Mapillary street-view photos using YOLOv8 and converts pixel-level detections into survey-grade 3D coordinates via multi-view triangulation.

---

## Pipeline Overview

```
Mapillary Vistas Dataset
        │
        ▼
 [1] Dataset Conversion
  Mapillary_Vistas_to_YOLO.ipynb
  • Polygon annotations → YOLO bounding boxes
  • NMS deduplication
  • 1,959 images, 35,000+ annotated objects
        │
        ▼
 [2] YOLOv8 Training
  • Trained on 8 utility classes
  • NVIDIA RTX A4500 (20GB)
  • 171 epochs
        │
        ▼
 [3] Live Inference on Project Sites
  Utility_Mapillary_API.ipynb
  • Pull street-level imagery via Mapillary API
  • Run YOLOv8 inference on real survey locations
        │
        ▼
 [4] Multi-View Triangulation
  Best_Triangulation.py
  • Parse camera poses from TopoDOT .lst files
  • Cast rays from each detection into 3D space
  • RANSAC refinement + DBSCAN clustering
  • Output: survey-grade XYZ coordinates + shapefiles
        │
        ▼
 [5] Accuracy Evaluation
  compare_utilities_eval.py
  • Compare predicted coordinates vs. ground-truth centroids
  • TP/FP/FN, Precision/Recall/F1 at distance tolerances (0.5 ft, 1.0 ft)
  • Per-class PR curves, FP breakdown, overlay plots
  • ~70% positional accuracy vs. ground-truth survey data
```

---

## Detection Model Results

Trained on the [Mapillary Vistas v2.0](https://www.mapillary.com/dataset/vistas) dataset. Validated on 4,517 images with 37,562 instances.

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 4517 | 37562 | 0.622 | 0.451 | 0.462 | 0.293 |
| fire_hydrant | 573 | 595 | 0.613 | 0.459 | 0.502 | 0.305 |
| junction_box | 1025 | 1229 | 0.606 | 0.493 | 0.486 | 0.349 |
| manhole | 1250 | 1823 | 0.659 | 0.470 | 0.508 | 0.340 |
| parking_meter | 574 | 629 | 0.542 | 0.480 | 0.461 | 0.300 |
| water_valve | 456 | 527 | 0.459 | 0.516 | 0.413 | 0.272 |
| traffic_sign | 1897 | 18456 | 0.703 | 0.360 | 0.416 | 0.266 |
| traffic_light | 991 | 7590 | 0.780 | 0.476 | 0.537 | 0.305 |
| utility_pole | 1063 | 6713 | 0.616 | 0.349 | 0.374 | 0.204 |

> Model: YOLOv8 (92 layers, 25.8M parameters) — Ultralytics 8.3.178, PyTorch 2.5.1+cu121

---

## Triangulation Approach

`Best_Triangulation.py` converts YOLO bounding box detections into real-world 3D coordinates using multi-view geometry:

- **Camera pose parsing** — reads position (XYZ) and rotation matrices from TopoDOT `.lst` files
- **Ray casting** — projects bounding box centroids into 3D rays using calibrated camera intrinsics (fx=fy=1024)
- **Pair-wise intersection** — finds ray midpoints across image pairs; filters by baseline distance (≥12 ft), intersection angle (≥3.5°), and pair distance (≤1.5 ft)
- **DBSCAN clustering** — groups intersection midpoints into candidate utility locations (eps=2.0 ft, min_samples=3)
- **RANSAC refinement** — robustly fits a 3D point per cluster; requires ≥4 inlier views and mean pair angle ≥5°
- **Touch filter** — final acceptance requires rays to pass within 0.35 ft of the estimated point
- **Output** — per-utility CSV of triangulated XYZ points + ESRI Shapefiles (PointZ, EPSG:6420)

Class-specific range gates prevent false matches (e.g. manholes capped at 80 ft, utility poles at 300 ft).

---

## Accuracy Evaluation

`compare_utilities_eval.py` evaluates triangulated predictions against ground-truth survey coordinates:

- **Spatial matching** — 1:1 nearest-neighbor matching via KDTree at configurable distance tolerances (0.5 ft, 1.0 ft)
- **Metrics** — TP, FP, FN, Precision, Recall, F1, median distance, P95 distance
- **FP breakdown** — classifies false positives as DUPLICATE, NEAR_MISS (≤2 ft), or FAR
- **PR curves** — confidence threshold sweep per class if confidence scores are present
- **Outputs** — `summary_overall.csv`, `summary_by_class.csv`, `matches_nearest.csv`, `overlay.png`

Validated against surveyor-collected ground truth centroids for fire hydrants, manholes, parking meters, traffic signs, and water valves on the Adeline project site, achieving **~70% positional accuracy** within tolerance.

---

## Repository Structure

```
├── Mapillary_Vistas_to_YOLO.ipynb   # Convert Mapillary Vistas → YOLO format
├── Utility_Mapillary_API.ipynb       # Pull live imagery from Mapillary API
├── Best_Triangulation.py             # Multi-view triangulation pipeline
├── compare_utilities_eval.py         # Accuracy evaluation vs. ground truth
├── Adeline Utility Accuracy/         # Per-class evaluation results
├── Adeline1-30 True Positions/       # Ground truth centroid CSVs
│   ├── FireHydrants_centroids.csv
│   ├── Manholes_centroids.csv
│   ├── ParkingMeter_centroids.csv
│   ├── TrafficSigns_centroids.csv
│   └── WaterValve_centroids.csv
└── RESULTS/                          # Triangulated output CSVs and shapefiles
```

---

## Setup

```bash
pip install ultralytics numpy pandas scikit-learn scipy matplotlib geopandas shapely fiona tqdm Pillow
```

**For triangulation**, camera poses must be exported from TopoDOT as `.lst` files containing `IMAGE=`, `XYZ=`, and `MAT=` fields per image block.

**For dataset conversion**, download [Mapillary Vistas v2.0](https://www.mapillary.com/dataset/vistas) and update the `ROOT` path in `Mapillary_Vistas_to_YOLO.ipynb`.

---

## Tech Stack

| Component | Tools |
|---|---|
| Object Detection | YOLOv8 (Ultralytics), PyTorch |
| Dataset Preparation | Mapillary Vistas v2.0, custom YOLO converter |
| Imagery Collection | Mapillary API |
| Triangulation | NumPy, SciPy (cKDTree), DBSCAN, RANSAC |
| Geospatial Output | GeoPandas, Shapely, EPSG:6420 |
| Ground Truth Validation | Custom KDTree spatial matcher |
| Survey Integration | TopoDOT, MicroStation |

---

## Context

Traditional utility surveys require field crews to manually locate and record infrastructure. This pipeline automates that process using existing street-level imagery, reducing the time and cost of utility identification for survey projects. The triangulated coordinates are accurate enough to serve as preliminary survey-grade estimates, with ~70% of detections falling within the acceptable tolerance of ground-truth positions.
