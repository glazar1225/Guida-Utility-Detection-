#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_utilities_eval.py

Compare predicted utilities vs ground-truth utilities from CSVs.

Features
- Auto-detect X/Y (and optional Z), class, and confidence columns
- 1:1 spatial matching under one or more distance tolerances (feet)
- Overall and per-class metrics: TP/FP/FN, Precision/Recall/F1, median and P95 distance
- Duplicate rate estimate and FP breakdown (DUPLICATE / NEAR_MISS / FAR)
- Optional confidence sweep (PR curve) if a confidence column exists
- Optional AOI bounding box to limit scoring region
- Robust KDTree/NearestNeighbors/brute-force fallbacks for matching

USAGE (Jupyter/Script):
1) Edit the CONFIG block below: PRED_CSV, GT_CSV, OUT_DIR, TOLERANCES_FT, etc.
2) In a Jupyter cell: 
       from compare_utilities_eval import run
       run()
   Or run as a script:
       python compare_utilities_eval.py

INPUT FORMAT
- CSV of predictions (PRED_CSV) and CSV of ground truth (GT_CSV)
- XY columns are auto-detected among: X/Y, E/N, Easting/Northing, lon/lat, etc.
- Optional columns:
    class:  "class", "Class", "feature", "Feature", "name", "Name"
    conf:   "conf", "confidence", "score", "prob", "Conf", "Confidence"
    id:     any id column is preserved if present but not required
- Units: XY must be in the same projected CRS (feet), not lat/lon in degrees

OUTPUTS (in OUT_DIR)
- matches_nearest.csv
- predictions_with_eval.csv
- groundtruth_with_eval.csv
- summary_overall.csv
- summary_by_class.csv              (if class columns exist)
- fp_breakdown_r{R}.csv            (for the largest tolerance R)
- pr_curve_by_class.csv            (if confidence exists)
- overlay.png                      (simple XY scatter for sanity check)

"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

# =====================
# CONFIG (EDIT THESE)
# =====================
PRED_CSV = r"G:\Users\hlienielsen\adeline_block1_labled\RESULTS\0_fire_hydrant\fire hydrant triangulated points.csv"   # <-- EDIT
GT_CSV   = r"G:\Users\hlienielsen\adeline_block1_labled\Adeline1-30 True Positions\FireHydrants_centroids.csv"   # <-- EDIT
OUT_DIR  = r"C:\Users\GLazar\Downloads\Utility Accuracy"                 # <-- EDIT (will be created)

# Distance tolerances in feet
TOLERANCES_FT = [0.5, 1.0]

# Optional: filter to an AOI bbox = (minX, minY, maxX, maxY) in same units as XY
AOI_BBOX: Optional[Tuple[float, float, float, float]] = None

# Optional: map prediction class names to ground-truth class names (if they differ)
# Example: {"traffic_sign":"sign", "traffic_light":"signal"}
CLASS_MAP: Dict[str, str] = {}

# Optional: confidence sweep steps if a confidence column exists
CONF_THRESHOLDS = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]  # 0.00..1.00 step 0.05

# Near-miss band upper bound (feet) for FP breakdown
NEAR_MISS_MAX_FT = 2.0

# ==============
# I/O utilities
# ==============
def _auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Heuristic to detect X, Y, optional Z, class, confidence columns."""
    cols = {c: c.strip() for c in df.columns}
    lower = {c.lower(): c for c in cols.values()}
    # XY candidates
    xy_pairs = [
        ("X","Y"), ("x","y"),
        ("E","N"), ("e","n"),
        ("Easting","Northing"), ("easting","northing"),
        ("lon","lat"), ("Lon","Lat"),
        ("longitude","latitude"), ("LONG","LAT")
    ]
    xcol = ycol = None
    for a,b in xy_pairs:
        if a in df.columns and b in df.columns:
            if pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
                xcol, ycol = a, b
                break
    if xcol is None or ycol is None:
        # fallback: first two numeric columns
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            xcol, ycol = num_cols[0], num_cols[1]
        else:
            raise ValueError("Could not auto-detect XY columns. Please rename or add X/Y.")

    zcol = None
    for cand in ["Z","z","elev","elevation","height"]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            zcol = cand
            break

    classcol = None
    for cand in ["class","Class","feature","Feature","name","Name","label","Label"]:
        if cand in df.columns:
            classcol = cand
            break

    confcol = None
    for cand in ["conf","confidence","score","prob","Conf","Confidence"]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            confcol = cand
            break

    return {"x": xcol, "y": ycol, "z": zcol, "cls": classcol, "conf": confcol}

def _load_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    meta = _auto_detect_columns(df)
    out = pd.DataFrame({
        "X": pd.to_numeric(df[meta["x"]], errors="coerce"),
        "Y": pd.to_numeric(df[meta["y"]], errors="coerce"),
    })
    if meta["z"]:
        out["Z"] = pd.to_numeric(df[meta["z"]], errors="coerce")
    if meta["cls"]:
        out["class"] = df[meta["cls"]].astype(str)
    if meta["conf"]:
        out["conf"] = pd.to_numeric(df[meta["conf"]], errors="coerce")
    out["row_id"] = df.index
    return out.dropna(subset=["X","Y"]).reset_index(drop=True), meta

def _apply_aoi(df: pd.DataFrame, bbox: Optional[Tuple[float,float,float,float]]) -> pd.DataFrame:
    if not bbox:
        return df
    minx, miny, maxx, maxy = bbox
    return df[(df["X"]>=minx) & (df["X"]<=maxx) & (df["Y"]>=miny) & (df["Y"]<=maxy)].reset_index(drop=True)

# =====================
# Matching primitives
# =====================
def _nearest_neighbors(Axy: np.ndarray, Bxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each A point, returns (idx_B, dist)."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(Bxy)
        dists, idxs = tree.query(Axy, k=1)
        return idxs, dists
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Bxy)
            dists, idxs = nn.kneighbors(Axy, n_neighbors=1, return_distance=True)
            return idxs.ravel(), dists.ravel()
        except Exception:
            nA, nB = Axy.shape[0], Bxy.shape[0]
            idxs = np.empty(nA, dtype=int)
            dists = np.empty(nA, dtype=float)
            chunk = 5000
            for start in range(0, nA, chunk):
                end = min(start+chunk, nA)
                sub = Axy[start:end]
                dx = sub[:, [0]] - Bxy[:, 0][None, :]
                dy = sub[:, [1]] - Bxy[:, 1][None, :]
                dist2 = dx*dx + dy*dy
                local_idxs = np.argmin(dist2, axis=1)
                local_d = np.sqrt(dist2[np.arange(dist2.shape[0]), local_idxs])
                idxs[start:end] = local_idxs
                dists[start:end] = local_d
            return idxs, dists

def _query_ball_counts(Axy: np.ndarray, Bxy: np.ndarray, r: float) -> np.ndarray:
    """Return count of B points within radius r for each A point."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(Bxy)
        return np.array([len(tree.query_ball_point(a, r)) for a in Axy], dtype=int)
    except Exception:
        r2 = r*r
        counts = np.zeros(Axy.shape[0], dtype=int)
        chunk = 4000
        for start in range(0, Axy.shape[0], chunk):
            end = min(start+chunk, Axy.shape[0])
            sub = Axy[start:end]
            dx = sub[:, [0]] - Bxy[:, 0][None, :]
            dy = sub[:, [1]] - Bxy[:, 1][None, :]
            dist2 = dx*dx + dy*dy
            counts[start:end] = (dist2 <= r2).sum(axis=1)
        return counts

def _one_to_one_match(Axy: np.ndarray, Bxy: np.ndarray, r: float) -> List[Tuple[int,int,float]]:
    """Greedy 1:1 matching using all candidate pairs with dist<=r, sorted by distance."""
    candidates = []
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(Bxy)
        for i, a in enumerate(Axy):
            js = tree.query_ball_point(a, r)
            if js:
                d = np.sqrt((Bxy[js,0]-a[0])**2 + (Bxy[js,1]-a[1])**2)
                for j, dist in zip(js, d):
                    candidates.append((i, int(j), float(dist)))
    except Exception:
        r2 = r*r
        for i, a in enumerate(Axy):
            dx = Bxy[:,0]-a[0]; dy = Bxy[:,1]-a[1]
            d2 = dx*dx + dy*dy
            js = np.where(d2<=r2)[0]
            for j in js:
                candidates.append((i, int(j), float(np.sqrt(d2[j]))))
    candidates.sort(key=lambda t: t[2])
    usedA=set(); usedB=set(); pairs=[]
    for i,j,d in candidates:
        if i not in usedA and j not in usedB:
            usedA.add(i); usedB.add(j); pairs.append((i,j,d))
    return pairs

# =====================
# Core evaluation
# =====================
def _eval_at_r(pred_df: pd.DataFrame, gt_df: pd.DataFrame, r: float) -> Dict:
    A = pred_df[["X","Y"]].to_numpy()
    B = gt_df[["X","Y"]].to_numpy()
    pairs = _one_to_one_match(A, B, r)
    tp = len(pairs)
    fp = A.shape[0] - tp
    fn = B.shape[0] - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = (2*prec*rec)/(prec+rec) if prec>0 and rec>0 else 0.0 if (prec==0 or rec==0) else float("nan")
    dists = np.array([d for _,_,d in pairs], dtype=float)
    md    = float(np.nanmedian(dists)) if dists.size else float("nan")
    p95   = float(np.nanpercentile(dists,95)) if dists.size else float("nan")

    # Duplicate estimate = (#pred with any GT within r) - tp
    has_neighbor = _query_ball_counts(A, B, r) > 0
    duplicates = int(has_neighbor.sum() - tp)
    dup_rate   = duplicates / A.shape[0] if A.shape[0] else float("nan")

    # Per-row status vectors
    pred_status = np.full(A.shape[0], "FP", dtype=object)
    pred_match_j = np.full(A.shape[0], -1, dtype=int)
    pred_match_d = np.full(A.shape[0], np.nan, dtype=float)
    gt_status   = np.full(B.shape[0], "FN", dtype=object)
    gt_match_i  = np.full(B.shape[0], -1, dtype=int)
    gt_match_d  = np.full(B.shape[0], np.nan, dtype=float)
    for i,j,d in pairs:
        pred_status[i] = "TP"
        pred_match_j[i] = j
        pred_match_d[i] = d
        gt_status[j] = "Matched"
        gt_match_i[j] = i
        gt_match_d[j] = d

    return {
        "pairs": pairs,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "median_dist": md, "p95_dist": p95,
        "duplicates": duplicates, "dup_rate": dup_rate,
        "pred_status": pred_status, "pred_match_j": pred_match_j, "pred_match_d": pred_match_d,
        "gt_status": gt_status, "gt_match_i": gt_match_i, "gt_match_d": gt_match_d
    }

def evaluate_utilities(
    pred_csv: str, gt_csv: str, out_dir: str,
    tolerances_ft: List[float] = None,
    aoi_bbox: Optional[Tuple[float,float,float,float]] = None,
    class_map: Dict[str,str] = None,
    conf_thresholds: Optional[List[float]] = None,
    near_miss_max_ft: float = 2.0,
    make_overlay: bool = True,
) -> Dict:
    """Main entry point. Returns paths to key outputs."""
    tolerances_ft = tolerances_ft or [0.5, 1.0]
    class_map = class_map or {}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    pred_df_raw, meta_pred = _load_csv(Path(pred_csv))
    gt_df_raw,   meta_gt   = _load_csv(Path(gt_csv))

    # Apply AOI if provided
    pred_df = _apply_aoi(pred_df_raw, aoi_bbox)
    gt_df   = _apply_aoi(gt_df_raw,   aoi_bbox)

    # Map class names if both have classes
    have_classes = ("class" in pred_df.columns) and ("class" in gt_df.columns)
    if have_classes and class_map:
        pred_df["class"] = pred_df["class"].map(lambda c: class_map.get(c, c))

    # Save nearest neighbors (overall)
    idx, dist = _nearest_neighbors(pred_df[["X","Y"]].to_numpy(), gt_df[["X","Y"]].to_numpy())
    nearest = pred_df.copy()
    nearest["nearest_gt_idx"] = idx
    nearest["nearest_dist"]   = dist
    nearest["gt_X"] = gt_df.loc[idx, "X"].to_numpy()
    nearest["gt_Y"] = gt_df.loc[idx, "Y"].to_numpy()
    if "Z" in gt_df.columns:
        nearest["gt_Z"] = gt_df.loc[idx, "Z"].to_numpy()
    nearest_out = str(out / "matches_nearest.csv")
    nearest.to_csv(nearest_out, index=False)

    # Overall + per-class loop
    def _eval_block(pd_block: pd.DataFrame, gt_block: pd.DataFrame, tag: str):
        summaries = []
        # clone for per-row outputs
        preds_eval = pd_block.copy()
        gts_eval   = gt_block.copy()
        for r in tolerances_ft:
            res = _eval_at_r(pd_block, gt_block, r)
            # write per-row status columns
            preds_eval[f"status_r{r}"]       = res["pred_status"]
            preds_eval[f"match_gt_idx_r{r}"] = res["pred_match_j"]
            preds_eval[f"match_dist_r{r}"]   = res["pred_match_d"]
            gts_eval[f"status_r{r}"]         = res["gt_status"]
            gts_eval[f"match_pred_idx_r{r}"] = res["gt_match_i"]
            gts_eval[f"match_dist_r{r}"]     = res["gt_match_d"]
            # summary row
            summaries.append({
                "tag": tag, "threshold_ft": r,
                "n_predictions": int(len(pd_block)),
                "n_groundtruth": int(len(gt_block)),
                "TP": res["tp"], "FP": res["fp"], "FN": res["fn"],
                "Precision": res["precision"], "Recall": res["recall"], "F1": res["f1"],
                "MedianDist": res["median_dist"], "P95Dist": res["p95_dist"],
                "Duplicates": res["duplicates"], "DupRate": res["dup_rate"],
            })
        return preds_eval, gts_eval, pd.DataFrame(summaries)

    # Overall
    preds_eval_all, gts_eval_all, summary_all = _eval_block(pred_df, gt_df, tag="ALL")
    preds_eval_all.to_csv(out / "predictions_with_eval.csv", index=False)
    gts_eval_all.to_csv(out / "groundtruth_with_eval.csv", index=False)
    summary_all.to_csv(out / "summary_overall.csv", index=False)

    # Per-class if available
    class_summaries = []
    if have_classes:
        classes = sorted(set(gt_df["class"]).union(set(pred_df["class"])))
        per_class_rows = []
        for c in classes:
            pd_c = pred_df[pred_df["class"]==c].reset_index(drop=True)
            gt_c = gt_df[gt_df["class"]==c].reset_index(drop=True)
            _, _, sum_c = _eval_block(pd_c, gt_c, tag=str(c))
            class_summaries.append(sum_c)
        if class_summaries:
            summary_by_class = pd.concat(class_summaries, ignore_index=True)
            summary_by_class.to_csv(out / "summary_by_class.csv", index=False)

    # FP breakdown at the largest tolerance
    R = max(tolerances_ft)
    A = preds_eval_all[["X","Y"]].to_numpy()
    B = gts_eval_all[["X","Y"]].to_numpy()
    pairs = _one_to_one_match(A, B, R)
    matched_pred = {i for i,_,_ in pairs}

    def _counts_within(Axy, Bxy, r):
        return _query_ball_counts(Axy, Bxy, r)

    within_R   = _counts_within(A, B, R)
    within_max = _counts_within(A, B, near_miss_max_ft)

    fp_type = []
    for i in range(A.shape[0]):
        if i in matched_pred:
            fp_type.append("TP")
        else:
            if within_R[i] > 0:
                fp_type.append("DUPLICATE")
            elif within_max[i] > 0:
                fp_type.append(f"NEAR_MISS_<= {near_miss_max_ft}ft")
            else:
                fp_type.append(f"FAR_> {near_miss_max_ft}ft")
    preds_eval_all["fp_breakdown_r{}".format(R)] = fp_type
    preds_eval_all.to_csv(out / "predictions_with_eval.csv", index=False)  # overwrite with breakdown col
    preds_eval_all[["fp_breakdown_r{}".format(R)]].value_counts().to_csv(out / f"fp_breakdown_r{R}.csv")

    # Confidence PR curve per class (if conf column exists)
    confcol_exists = "conf" in pred_df.columns
    if confcol_exists and have_classes and conf_thresholds:
        rows = []
        for c in classes:
            pd_c_full = pred_df[pred_df["class"]==c]
            gt_c      = gt_df[gt_df["class"]==c]
            for t in conf_thresholds:
                pd_c = pd_c_full[pd_c_full["conf"]>=t].reset_index(drop=True)
                res  = _eval_at_r(pd_c, gt_c, R)
                rows.append({
                    "class": c, "conf_threshold": t, "threshold_ft": R,
                    "n_predictions": int(len(pd_c)), "n_groundtruth": int(len(gt_c)),
                    "TP": res["tp"], "FP": res["fp"], "FN": res["fn"],
                    "Precision": res["precision"], "Recall": res["recall"], "F1": res["f1"],
                })
        pd.DataFrame(rows).to_csv(out / "pr_curve_by_class.csv", index=False)

    # Quick overlay plot
    if make_overlay:
        try:
            import matplotlib.pyplot as plt
            max_points = 5000
            pd_plot = pred_df.sample(n=min(len(pred_df), max_points), random_state=0) if len(pred_df) > max_points else pred_df
            gt_plot = gt_df.sample(n=min(len(gt_df), max_points), random_state=1) if len(gt_df) > max_points else gt_df
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6), dpi=150)
            plt.scatter(gt_plot["X"], gt_plot["Y"], s=6, label="Ground Truth")
            plt.scatter(pd_plot["X"], pd_plot["Y"], s=6, marker="x", label="Predictions")
            plt.legend()
            plt.title("Predictions vs Ground Truth (XY)")
            plt.xlabel("X"); plt.ylabel("Y")
            plt.tight_layout()
            plt.savefig(out / "overlay.png", bbox_inches="tight")
            plt.close()
        except Exception as e:
            print("Overlay plot failed:", e)

    return {
        "nearest_out": str(out / "matches_nearest.csv"),
        "predictions_with_eval_out": str(out / "predictions_with_eval.csv"),
        "groundtruth_with_eval_out": str(out / "groundtruth_with_eval.csv"),
        "summary_overall_out": str(out / "summary_overall.csv"),
        "summary_by_class_out": str(out / "summary_by_class.csv") if have_classes else None,
        "fp_breakdown_out": str(out / f"fp_breakdown_r{R}.csv"),
        "pr_curve_by_class_out": str(out / "pr_curve_by_class.csv") if confcol_exists and have_classes and conf_thresholds else None,
        "overlay_png": str(out / "overlay.png") if make_overlay else None,
    }

# =====================
# Jupyter-friendly run
# =====================
def run():
    return evaluate_utilities(
        pred_csv=PRED_CSV,
        gt_csv=GT_CSV,
        out_dir=OUT_DIR,
        tolerances_ft=TOLERANCES_FT,
        aoi_bbox=AOI_BBOX,
        class_map=CLASS_MAP,
        conf_thresholds=CONF_THRESHOLDS,
        near_miss_max_ft=NEAR_MISS_MAX_FT,
        make_overlay=True,
    )

if __name__ == "__main__":
    print("Running with CONFIG at top of file...")
    paths = run()
    for k,v in paths.items():
        print(f"{k}: {v}")
