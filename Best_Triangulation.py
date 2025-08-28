#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
triangulate_multi_utilities_fast.py

• Inputs picked via GUI:
  - IMAGES root (all photos)
  - LABELS ROOT (folder of utility subfolders; each contains YOLO .txt predictions with confidences)
  - LST FOLDER (many .lst files)
  - OUTPUT root

• Zero-accuracy-loss optimizations:
  - Caches image sizes once (no repeated disk reads).
  - Resolves rotation once from first utility that has samples, then reuses for all.
  - Uses scikit-learn DBSCAN if installed (faster); otherwise a pure-Python fallback.

• Behavior (unchanged mathematically):
  - STRICT confidence filter: only 6-token rows (class x y w h conf), conf ≥ CONF_MIN.
  - Auto-rotation (R vs Rᵀ) and ±Z guess using pair geometry.
  - Class-specific range gates by utility folder name (if it maps to NAMES), else defaults.
  - Strong geometry + touch filter.
  - Per-utility outputs named: "<utility pretty> triangulated points.csv/.shp" (PointZ).

Tip: Run YOLO inference with save_txt=True and save_conf=True to get confidences in the label .txt files.
"""

from __future__ import annotations
import math, re, csv, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# ===================== Intrinsics & thresholds =====================
FX = 1024.0
FY = 1024.0
CX = 1024.0
CY = 1024.0
ASSUMED_IMG_W = 2048
ASSUMED_IMG_H = 2048
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Labels: STRICT 6-token rows only (class x y w h conf)
CONF_MIN = 0.50

# Pair gating & clustering (feet / degrees)
MIN_BASELINE_FT     = 12.0
MIN_PAIR_ANGLE_DEG  = 3.5
PAIR_MAX_DIST_FT    = 1.5
CLUSTER_EPS_FT      = 2.0
CLUSTER_MIN_SAMPLES = 3
MAX_VIEWS           = 8
INLIER_DIST_FT      = 1.5
QUALITY_DROP        = 0.90

# Strong acceptance
MIN_INLIER_VIEWS    = 4
MIN_MEAN_ANGLE_DEG  = 5.0

# Touch filter
TOUCH_MAX_DIST_FT   = 0.35
MIN_TOUCH_COUNT     = 1
MIN_TOUCH_ANGLE_DEG = 3.0  # only used if MIN_TOUCH_COUNT>=2

# -------- Range gates (feet) --------
DEFAULT_MIN_RANGE_FT = 8.0
DEFAULT_MAX_RANGE_FT = 250.0
MAX_RANGE_BY_CLASS_FT = {
    "traffic_sign":   180.0,
    "traffic_light":  240.0,
    "manhole":         80.0,
    "junction_box":    90.0,
    "water_valve":     90.0,
    "parking_meter":  120.0,
    "utility_pole":   300.0,
    "fire_hydrant":   120.0,
}
def get_range_limits_by_name(name_norm: str) -> Tuple[float,float]:
    return DEFAULT_MIN_RANGE_FT, MAX_RANGE_BY_CLASS_FT.get(name_norm, DEFAULT_MAX_RANGE_FT)

# Optional sparse 2-view fallback (disabled by default)
SPARSE_ENABLE          = False
SPARSE_MIN_ANGLE_DEG   = 6.0
SPARSE_MIN_BASELINE_FT = 15.0
SPARSE_MAX_GAP_FT      = 0.45
SPARSE_QUALITY_DROP    = 0.90

# Memory-safety caps
MAX_PAIRS_PER_RAY   = 40
MAX_PAIRPOINTS      = 200_000

# Class names (adjust to your model if needed)
NAMES = [
    "fire_hydrant",
    "junction_box",
    "manhole",
    "parking_meter",
    "water_valve",
    "traffic_sign",
    "traffic_light",
    "utility_pole",
]

# Rotation auto-resolution (reused globally)
ROT_MODE         = "auto"     # "R", "RT", or "auto"
AUTO_SAMPLE_MAX  = 400
ROT_USE_RT: bool = False
ROT_Z_SIGN: int  = +1
ROT_RESOLVED: bool = False

# ===================== Optional deps =====================
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    import fiona
    HAVE_GPD = True
except Exception:
    HAVE_GPD = False

# ===================== GUI pickers =====================
import tkinter as tk
from tkinter import filedialog, messagebox

def pick_dir(title: str, mustexist=True) -> Path:
    p = filedialog.askdirectory(title=title, mustexist=mustexist)
    if not p: raise SystemExit("Canceled.")
    return Path(p)

# ===================== LST parser (multi-file) =====================
@dataclass
class Pose:
    C: np.ndarray  # camera center (world)
    R: np.ndarray  # rotation matrix from LST block

def parse_lst_blocks(lst_path: Path) -> Dict[str, Pose]:
    txt = lst_path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", txt.strip())
    poses: Dict[str, Pose] = {}
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines: continue
        image_line = next((l for l in lines if l.lower().startswith("image=")), None)
        xyz_line   = next((l for l in lines if l.lower().startswith("xyz=")), None)
        mat_line   = next((l for l in lines if l.lower().startswith("mat=")), None)
        if not (image_line and xyz_line and mat_line): continue

        image_name = image_line.split("=", 1)[1].strip().strip('"').strip("'")
        stem = Path(image_name).stem.lower()

        xyz_vals = [float(x) for x in xyz_line.split("=",1)[1].split()]
        if len(xyz_vals) < 3: continue
        C = np.array(xyz_vals[:3], float)

        mv = [float(x) for x in mat_line.split("=",1)[1].split()]
        if len(mv) != 9:
            mv = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", mat_line)]
            if len(mv) < 9: continue
            mv = mv[:9]
        R = np.array(mv, float).reshape(3,3)

        poses[stem] = Pose(C=C, R=R)
    return poses

def parse_all_lsts(lst_dir: Path) -> Dict[str, Pose]:
    merged: Dict[str, Pose] = {}
    lsts = sorted(lst_dir.glob("*.lst"))
    if not lsts:
        raise SystemExit(f"No .lst files found in {lst_dir}")
    for p in lsts:
        merged.update(parse_lst_blocks(p))  # later files override duplicates
    if not merged:
        raise SystemExit("No valid pose blocks parsed from any .lst file.")
    print(f"Parsed poses from {len(lsts)} LST files → unique images: {len(merged)}")
    return merged

# ===================== Image index + size cache =====================
IMAGE_INDEX: Dict[str, Path] = {}
IMAGE_WH_CACHE: Dict[str, Tuple[int,int]] = {}

def build_image_index(images_dir: Path):
    """Scan once; build {stem_lower: Path} and size cache."""
    IMAGE_INDEX.clear()
    IMAGE_WH_CACHE.clear()
    with os.scandir(images_dir) as it:
        for e in it:
            if not e.is_file(): continue
            ext = os.path.splitext(e.name)[1].lower()
            if ext in IMG_EXTS:
                stem = os.path.splitext(e.name)[0].lower()
                p = Path(e.path)
                IMAGE_INDEX[stem] = p
                # cache size
                if HAVE_CV2:
                    im = cv2.imread(str(p))
                    if im is not None:
                        h, w = im.shape[:2]
                        IMAGE_WH_CACHE[stem] = (w, h)
                        continue
                IMAGE_WH_CACHE[stem] = (ASSUMED_IMG_W, ASSUMED_IMG_H)
    print(f"Image index built: {len(IMAGE_INDEX)} files (sizes cached).")

# ===================== Labels & utilities =====================
@dataclass
class Det:
    key: str
    cls_id: int
    conf: float
    x_norm: float
    y_norm: float

def read_label_file_strict(lbl_path: Path) -> List[Tuple[int,float,float,float,float,float]]:
    """Return list of (class, conf, x, y, w, h) — STRICT 6-token rows only."""
    out = []
    for ln in lbl_path.read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = ln.split()
        if len(parts) != 6:
            continue  # STRICT: must include confidence
        cls, x, y, w, h, conf = map(float, parts)
        out.append((int(cls), float(conf), float(x), float(y), float(w), float(h)))
    return out

def find_label_dir_for_utility(util_folder: Path) -> Optional[Path]:
    """Prefer ./labels subdir if it contains .txt; else the folder itself; else first child with .txt."""
    labels_sub = util_folder / "labels"
    if labels_sub.exists() and any(labels_sub.glob("*.txt")):
        return labels_sub
    if any(util_folder.glob("*.txt")):
        return util_folder
    for child in util_folder.iterdir():
        if child.is_dir() and any(child.glob("*.txt")):
            return child
    return None

def normalize_util_name(folder_name: str) -> Tuple[str, Optional[int]]:
    """('0_fire_hydrant' -> 'fire_hydrant', 0)."""
    base = folder_name.strip()
    m = re.match(r"^\s*(\d+)[_\-\s]+(.+)$", base)
    if m:
        base = m.group(2)
    nm = base.lower()
    nm = re.sub(r"[^\w\s\-]+", "", nm)
    nm = re.sub(r"[\s\-]+", "_", nm).strip("_")
    if nm in NAMES:
        return nm, NAMES.index(nm)
    if nm.endswith("s") and nm[:-1] in NAMES:
        return nm[:-1], NAMES.index(nm[:-1])
    return nm, None

def collect_aligned_for_labels_dir(
    labels_dir: Path,
    poses: Dict[str, Pose],
    conf_min: float = CONF_MIN
) -> Tuple[List[Det], Dict[str,Path], Dict[str,Path], Optional[float], List[Tuple[np.ndarray,np.ndarray,float,float,int,int]]]:
    """Uses global IMAGE_INDEX and IMAGE_WH_CACHE (built once)."""
    lbl_map: Dict[str, Path] = {p.stem.lower(): p for p in labels_dir.glob("*.txt")}
    keys = sorted(set(poses) & set(IMAGE_INDEX) & set(lbl_map))

    dets: List[Det] = []
    min_kept_conf = None
    auto_samples: List[Tuple[np.ndarray,np.ndarray,float,float,int,int]] = []  # (C,R,xn,yn,w,h)

    for k in keys:
        p = poses[k]
        w, h = IMAGE_WH_CACHE.get(k, (ASSUMED_IMG_W, ASSUMED_IMG_H))
        for cls, conf, xn, yn, wn, hn in read_label_file_strict(lbl_map[k]):
            if conf < conf_min:
                continue
            dets.append(Det(key=k, cls_id=cls, conf=conf, x_norm=xn, y_norm=yn))
            if (min_kept_conf is None) or (conf < min_kept_conf):
                min_kept_conf = conf
            if len(auto_samples) < AUTO_SAMPLE_MAX:
                auto_samples.append((p.C, p.R, xn, yn, w, h))

    print(f"Aligned keys: {len(keys)} | Detections kept (conf≥{conf_min}): {len(dets)}")
    if dets and min_kept_conf is not None:
        print(f"   Min kept confidence: {min_kept_conf:.3f}")
    # We also return IMAGE_INDEX subset for convenience (same interface as before)
    img_map = {k: IMAGE_INDEX[k] for k in keys}
    return dets, img_map, lbl_map, min_kept_conf, auto_samples

# ===================== Rotation auto-resolution =====================
def pair_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cs = float(np.clip(a @ b, -1.0, 1.0))
    return math.degrees(math.acos(cs))

def _dir_from_hypo(C: np.ndarray, R: np.ndarray, xn: float, yn: float, w: int, h: int, use_RT: bool, z_sign: int) -> np.ndarray:
    fx_eff = FX * (w / 2048.0); fy_eff = FY * (h / 2048.0)
    cx_eff = CX * (w / 2048.0); cy_eff = CY * (h / 2048.0)
    u = xn * w; v = yn * h
    x = (u - cx_eff) / fx_eff; y = (v - cy_eff) / fy_eff
    vc = np.array([x, y, float(z_sign)], float)
    d = (R.T if use_RT else R) @ vc
    d = d / np.linalg.norm(d)
    return d

def score_rotation_hypothesis(samples, use_RT: bool, z_sign: int) -> int:
    if len(samples) < 2: return 0
    dirs = [(_dir_from_hypo(C,R,xn,yn,w,h,use_RT,z_sign), C) for (C,R,xn,yn,w,h) in samples]
    hits = 0
    for i in range(len(dirs)):
        d1, C1 = dirs[i]
        for j in range(i+1, len(dirs)):
            d2, C2 = dirs[j]
            if np.linalg.norm(C1 - C2) < MIN_BASELINE_FT: continue
            ang = pair_angle_deg(d1, d2)
            if ang < MIN_PAIR_ANGLE_DEG: continue
            w0 = C1 - C2
            a = d1 @ d1; b = d1 @ d2; c = d2 @ d2
            d = d1 @ w0; e = d2 @ w0
            denom = a*c - b*b
            if abs(denom) < 1e-9: continue
            t1 = (b*e - c*d) / denom
            t2 = (a*e - b*d) / denom
            gap = np.linalg.norm((C1 + t1*d1) - (C2 + t2*d2))
            if gap <= PAIR_MAX_DIST_FT: hits += 1
    return hits

def resolve_rotation_once(auto_samples):
    """Resolve rotation globally only once."""
    global ROT_USE_RT, ROT_Z_SIGN, ROT_RESOLVED
    if ROT_RESOLVED:
        print(f"Reusing rotation: use_RT={ROT_USE_RT}, z_sign={ROT_Z_SIGN}")
        return
    if ROT_MODE == "R":
        ROT_USE_RT, ROT_Z_SIGN, ROT_RESOLVED = False, +1, True
        print("Rotation mode: forced R, z_sign=+1")
        return
    if ROT_MODE == "RT":
        ROT_USE_RT, ROT_Z_SIGN, ROT_RESOLVED = True, +1, True
        print("Rotation mode: forced RT, z_sign=+1")
        return
    hypos = [(False, +1), (True, +1), (False, -1), (True, -1)]
    best_hits, best = -1, (False, +1)
    for use_rt, zsgn in hypos:
        hits = score_rotation_hypothesis(auto_samples, use_rt, zsgn)
        if hits > best_hits:
            best_hits, best = hits, (use_rt, zsgn)
    ROT_USE_RT, ROT_Z_SIGN = best
    ROT_RESOLVED = True
    print(f"Auto-rotation pick: use_RT={ROT_USE_RT}, z_sign={ROT_Z_SIGN} (pair hits={best_hits})")

# ===================== Rays & geometry =====================
@dataclass
class Ray:
    C: np.ndarray
    d: np.ndarray
    key: str
    conf: float
    cls_id: int

def pixel_to_ray_from_norm(xn: float, yn: float, pose: Pose, w: int, h: int) -> np.ndarray:
    fx_eff = FX * (w / 2048.0); fy_eff = FY * (h / 2048.0)
    cx_eff = CX * (w / 2048.0); cy_eff = CY * (h / 2048.0)
    u = xn * w; v = yn * h
    x = (u - cx_eff) / fx_eff; y = (v - cy_eff) / fy_eff
    vc = np.array([x, y, float(ROT_Z_SIGN)], float)
    Ruse = pose.R.T if ROT_USE_RT else pose.R
    d = Ruse @ vc
    d = d / np.linalg.norm(d)
    return d

def form_rays(dets: List[Det], poses: Dict[str, Pose]) -> List[Ray]:
    out: List[Ray] = []
    for d in dets:
        p = poses.get(d.key)
        if p is None: continue
        w, h = IMAGE_WH_CACHE.get(d.key, (ASSUMED_IMG_W, ASSUMED_IMG_H))
        direction = pixel_to_ray_from_norm(d.x_norm, d.y_norm, p, w, h)
        out.append(Ray(C=p.C, d=direction, key=d.key, conf=d.conf, cls_id=d.cls_id))
    return out

@dataclass
class PairIntersect:
    Pm: np.ndarray
    gap: float
    ti: float
    tj: float
    angle_deg: float

def intersect_pair(ri: Ray, rj: Ray) -> PairIntersect:
    p1, d1 = ri.C, ri.d
    p2, d2 = rj.C, rj.d
    w0 = p1 - p2
    a = d1 @ d1; b = d1 @ d2; c = d2 @ d2
    d = d1 @ w0; e = d2 @ w0
    denom = a*c - b*b
    if abs(denom) < 1e-9:
        ti = tj = 0.0
    else:
        ti = (b*e - c*d) / denom
        tj = (a*e - b*d) / denom
    qi = p1 + ti * d1
    qj = p2 + tj * d2
    Pm = 0.5 * (qi + qj)
    gap = float(np.linalg.norm(qi - qj))
    return PairIntersect(Pm=Pm, gap=gap, ti=float(ti), tj=float(tj), angle_deg=pair_angle_deg(d1,d2))

# ===================== Memory-safe clustering =====================
def grid_dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    N = pts.shape[0]
    if N == 0: return np.empty((0,), dtype=int)
    inv = 1.0 / float(eps)
    vox = np.floor(pts * inv).astype(np.int64)
    cell_map: Dict[Tuple[int,int,int], List[int]] = {}
    for i, c in enumerate(vox):
        key = (int(c[0]), int(c[1]), int(c[2]))
        cell_map.setdefault(key, []).append(i)
    neigh = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
    eps2 = float(eps*eps)
    def region_query(i: int) -> List[int]:
        cx, cy, cz = vox[i]
        pi = pts[i]
        out = []
        for dx,dy,dz in neigh:
            key = (int(cx+dx), int(cy+dy), int(cz+dz))
            if key not in cell_map: continue
            for j in cell_map[key]:
                d2 = float((pts[j][0]-pi[0])**2 + (pts[j][1]-pi[1])**2 + (pts[j][2]-pi[2])**2)
                if d2 <= eps2: out.append(j)
        return out
    labels = np.full(N, -1, dtype=int)
    visited = np.zeros(N, dtype=bool)
    cid = 0
    for i in range(N):
        if visited[i]: continue
        visited[i] = True
        neigh_i = region_query(i)
        if len(neigh_i) < min_samples: continue
        labels[i] = cid
        seeds = [j for j in neigh_i if j != i]
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = region_query(j)
                if len(neigh_j) >= min_samples:
                    seeds.extend([n for n in neigh_j if labels[n] == -1])
            if labels[j] == -1:
                labels[j] = cid
        cid += 1
    return labels

def cluster_pair_points(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    pts = points.astype(np.float32, copy=False)
    try:
        from sklearn.cluster import DBSCAN as SK_DBSCAN
        return SK_DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                         algorithm='ball_tree', n_jobs=-1).fit_predict(pts)
    except Exception:
        return grid_dbscan(pts, eps, min_samples)

# ===================== Triangulation helpers =====================
def triangulate_ls(rays: List[Ray]) -> np.ndarray:
    A = np.zeros((3,3)); b = np.zeros(3)
    I = np.eye(3)
    for r in rays:
        P = I - np.outer(r.d, r.d)
        A += P; b += P @ r.C
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, b, rcond=None)[0]
    return X

def ray_point_distance(ray: Ray, P: np.ndarray) -> float:
    return float(np.linalg.norm(np.cross(ray.d, P - ray.C)))

def along_ray_t(ray: Ray, P: np.ndarray) -> float:
    return float(np.dot(P - ray.C, ray.d))  # feet (d is unit)

def select_views_max_angle(rays: List[Ray], max_keep: int, min_pair_angle_deg: float) -> List[Ray]:
    if len(rays) <= max_keep: return rays
    m = len(rays)
    ang = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            a = pair_angle_deg(rays[i].d, rays[j].d)
            ang[i,j]=ang[j,i]=a
    i0,j0 = np.unravel_index(np.argmax(ang), ang.shape)
    sel = {i0,j0}
    while len(sel) < max_keep:
        best_k, best_gain = None, -1.0
        for k in range(m):
            if k in sel: continue
            mins = [ang[k,s] for s in sel]
            if max(mins) < min_pair_angle_deg: continue
            gain = min(mins)
            if gain > best_gain:
                best_gain = gain; best_k = k
        if best_k is None: break
        sel.add(best_k)
    return [rays[i] for i in sorted(sel)]

def ransac_refine(rays: List[Ray], inlier_dist_ft: float, min_keep: int=3) -> Tuple[Optional[np.ndarray], List[int]]:
    if len(rays) < min_keep: return None, []
    P = triangulate_ls(rays)
    d = np.array([ray_point_distance(r, P) for r in rays])
    inliers = np.where(d < inlier_dist_ft)[0].tolist()
    if len(inliers) < min_keep: return None, []
    P2 = triangulate_ls([r for i,r in enumerate(rays) if i in inliers])
    d2 = np.array([ray_point_distance(rays[i], P2) for i in inliers])
    inliers2 = [inliers[i] for i in range(len(inliers)) if d2[i] < inlier_dist_ft]
    if len(inliers2) < min_keep: return None, []
    Pref = triangulate_ls([rays[i] for i in inliers2])
    return Pref, inliers2

# ===================== Per-utility pipeline =====================
def process_one_utility(
    util_name_folder: str,
    labels_dir: Path,
    poses: Dict[str, Pose],
    out_root: Path,
    viz_root: Optional[Path] = None
) -> int:
    # Normalize name and try map to NAMES for range limits
    name_norm, cls_idx_or_none = normalize_util_name(util_name_folder)
    name_pretty = name_norm.replace("_", " ")
    rng_min, rng_max = get_range_limits_by_name(name_norm)

    print("\n" + "="*80)
    print(f"UTILITY: {name_pretty}  (range gate: {rng_min}-{rng_max} ft)")
    print("="*80)

    dets, img_map, lbl_map, min_kept_conf, auto_samples = collect_aligned_for_labels_dir(labels_dir, poses, conf_min=CONF_MIN)

    # Resolve rotation ONCE (from the first utility that has samples)
    if auto_samples and len(auto_samples) >= 2 and (not ROT_RESOLVED):
        resolve_rotation_once(auto_samples)
    elif ROT_RESOLVED:
        print(f"Reusing rotation: use_RT={ROT_USE_RT}, z_sign={ROT_Z_SIGN}")
    else:
        print("Not enough samples to resolve rotation yet; will try later utilities.")

    # If folder implies a known class, override det.cls_id (so gates/names match)
    if cls_idx_or_none is not None:
        for d in dets:
            d.cls_id = cls_idx_or_none

    rays_all = form_rays(dets, poses)
    print(f"Rays formed: {len(rays_all)}")

    # Filter rays to this utility class if known; otherwise keep all (fallback ranges)
    if cls_idx_or_none is not None:
        rays = [r for r in rays_all if r.cls_id == cls_idx_or_none]
    else:
        rays = rays_all

    rows_out = []
    fid = 1

    # Build pair midpoints with gating + caps + RANGE GATE on ti/tj
    pair_pts = []
    pairs: List[Tuple[int,int,float]] = []
    counts = np.zeros(len(rays), dtype=np.int32)

    for i in range(len(rays)):
        if counts[i] >= MAX_PAIRS_PER_RAY:
            continue
        ri = rays[i]
        for j in range(i+1, len(rays)):
            if counts[j] >= MAX_PAIRS_PER_RAY:
                continue
            rj = rays[j]

            base = float(np.linalg.norm(ri.C - rj.C))
            if base < MIN_BASELINE_FT:
                continue

            ang = pair_angle_deg(ri.d, rj.d)
            if ang < MIN_PAIR_ANGLE_DEG:
                continue

            pi = intersect_pair(ri, rj)
            if pi.gap > PAIR_MAX_DIST_FT:
                continue

            # -------- RANGE GATE --------
            if not (rng_min <= pi.ti <= rng_max and rng_min <= pi.tj <= rng_max):
                continue

            pair_pts.append(np.asarray(pi.Pm, dtype=np.float32))
            pairs.append((i, j, ang))
            counts[i] += 1
            counts[j] += 1
            if len(pair_pts) >= MAX_PAIRPOINTS:
                break
        if len(pair_pts) >= MAX_PAIRPOINTS:
            break

    pair_pts = np.vstack(pair_pts) if len(pair_pts) else np.empty((0,3), dtype=np.float32)
    print(f"[{name_pretty}] valid pair midpoints: {len(pair_pts)}")
    if len(pair_pts) < CLUSTER_MIN_SAMPLES:
        # Optional sparse 2-view fallback with range gate
        if SPARSE_ENABLE and len(pairs) > 0:
            for (i, j, ang) in pairs:
                ri, rj = rays[i], rays[j]
                base = float(np.linalg.norm(ri.C - rj.C))
                pi = intersect_pair(ri, rj)
                if ang < SPARSE_MIN_ANGLE_DEG:     continue
                if base < SPARSE_MIN_BASELINE_FT:  continue
                if pi.gap > SPARSE_MAX_GAP_FT:     continue
                if not (rng_min <= pi.ti <= rng_max and rng_min <= pi.tj <= rng_max):
                    continue
                Pref = pi.Pm
                d1 = ray_point_distance(ri, Pref)
                d2 = ray_point_distance(rj, Pref)
                if max(d1, d2) > TOUCH_MAX_DIST_FT:
                    continue
                mean_angle = ang
                mean_res   = float((d1 + d2) / 2.0)
                geom = 1.0 - math.exp(-(mean_angle/5.0))
                fit  = math.exp(-(mean_res / 1.0))
                score = float(max(0.0, min(1.0, 0.5*geom + 0.5*fit)))
                if score < SPARSE_QUALITY_DROP:
                    continue
                conf_mean = float((ri.conf + rj.conf)/2.0)
                rows_out.append({
                    "feature_id": fid,
                    "class_name": name_pretty,
                    "x": Pref[0], "y": Pref[1], "z": Pref[2],
                    "views": 2,
                    "intersections": 1,
                    "mean_res_ft": round(mean_res, 3),
                    "mean_pair_angle_deg": round(mean_angle, 2),
                    "triangulation_score": round(score, 2),
                    "conf_mean": round(conf_mean, 3),
                    "touch_count": 2
                })
                fid += 1
        return write_outputs(rows_out, out_root, name_pretty)

    # Cluster & refine
    labels = cluster_pair_points(pair_pts, eps=CLUSTER_EPS_FT, min_samples=CLUSTER_MIN_SAMPLES)
    ncl = int(labels.max()) + 1 if labels.size else 0
    if ncl == 0:
        return write_outputs(rows_out, out_root, name_pretty)

    for c in range(ncl):
        idxs = np.where(labels == c)[0].tolist()
        if len(idxs) < CLUSTER_MIN_SAMPLES: continue

        # Rays that formed those pairs
        ray_ids: Set[int] = set()
        for k in idxs:
            i, j, _ = pairs[k]
            ray_ids.add(i); ray_ids.add(j)

        rays_feat = [rays[i] for i in sorted(ray_ids)]
        rays_feat = select_views_max_angle(rays_feat, max_keep=MAX_VIEWS, min_pair_angle_deg=MIN_PAIR_ANGLE_DEG)
        if len(rays_feat) < 3:
            continue

        Pref, inliers_idx = ransac_refine(rays_feat, inlier_dist_ft=INLIER_DIST_FT, min_keep=3)
        if Pref is None:
            continue

        inliers = [rays_feat[i] for i in inliers_idx]
        if len(inliers) < MIN_INLIER_VIEWS:
            continue

        # Mean pair angle among inliers
        angles = []
        for i in range(len(inliers)):
            for j in range(i+1, len(inliers)):
                angles.append(pair_angle_deg(inliers[i].d, inliers[j].d))
        mean_angle = float(np.mean(angles)) if angles else 0.0
        if mean_angle < MIN_MEAN_ANGLE_DEG:
            continue

        # Fit quality & score
        dists = [ray_point_distance(r, Pref) for r in inliers]
        mean_res = float(np.mean(dists))
        geom = 1.0 - math.exp(-(mean_angle/5.0))
        fit  = math.exp(-(mean_res / 1.5))
        score = float(max(0.0, min(1.0, 0.5*geom + 0.5*fit)))
        conf_mean = float(np.mean([r.conf for r in inliers]))
        if score < QUALITY_DROP:
            continue

        # Touch + range on touches
        touch_idx = [k for k, dist in enumerate(dists) if dist <= TOUCH_MAX_DIST_FT]
        if len(touch_idx) < MIN_TOUCH_COUNT:
            continue
        touch_ok = 0
        for k in touch_idx:
            r = inliers[k]
            ti = along_ray_t(r, Pref)
            if rng_min <= ti <= rng_max:
                touch_ok += 1
        if touch_ok < MIN_TOUCH_COUNT:
            continue
        if MIN_TOUCH_COUNT >= 2 and len(touch_idx) >= 2:
            touch_rays = [inliers[k] for k in touch_idx]
            max_touch_ang = 0.0
            for a in range(len(touch_rays)):
                for b in range(a+1, len(touch_rays)):
                    max_touch_ang = max(max_touch_ang, pair_angle_deg(touch_rays[a].d, touch_rays[b].d))
            if max_touch_ang < MIN_TOUCH_ANGLE_DEG:
                continue

        rows_out.append({
            "feature_id": fid,
            "class_name": name_pretty,
            "x": Pref[0], "y": Pref[1], "z": Pref[2],
            "views": len(inliers),
            "intersections": len(idxs),
            "mean_res_ft": round(mean_res, 3),
            "mean_pair_angle_deg": round(mean_angle, 2),
            "triangulation_score": round(score, 2),
            "conf_mean": round(conf_mean, 3),
            "touch_count": touch_ok
        })
        fid += 1

    return write_outputs(rows_out, out_root, name_pretty)

# ===================== Writers =====================
def write_outputs(rows_out: List[dict], out_root: Path, name_pretty: str) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    base = f"{name_pretty} triangulated points"
    csv_path = out_root / f"{base}.csv"
    fields = ["feature_id","class_name","x","y","z","views",
              "intersections","mean_res_ft","mean_pair_angle_deg","triangulation_score","conf_mean","touch_count"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows_out: w.writerow(r)
    print(f"✅ CSV: {csv_path}  (features: {len(rows_out)})")

    if HAVE_GPD and rows_out:
        pts3d = [Point(float(r["x"]), float(r["y"]), float(r["z"])) for r in rows_out]
        gdf = gpd.GeoDataFrame(rows_out, geometry=pts3d, crs="EPSG:6420")
        shp_path = out_root / f"{base}.shp"
        try:
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            print(f"✅ Shapefile (PointZ): {shp_path}")
        except Exception as e:
            print(f"⚠️ Shapefile write failed: {e}")
            gpkg_path = out_root / f"{base}.gpkg"
            gdf.to_file(gpkg_path, driver="GPKG")
            print(f"✅ GeoPackage (3D): {gpkg_path}")
    elif not HAVE_GPD:
        print("ℹ️ geopandas not installed — wrote CSV only.")
    return len(rows_out)

# ===================== Main =====================
def main():
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo(
        "Triangulation (multi-utility, fast)",
        "Pick:\n• IMAGES root\n• LABELS ROOT (folder of folders)\n• LST folder (many .lst)\n• OUTPUT root\n\n"
        f"STRICT: 6-token labels with confidences, conf ≥ {CONF_MIN}.\n"
        "Auto-rotation + range gates + strong geometry + touch filter."
    )

    images_dir = pick_dir("Pick IMAGES root")
    labels_root = pick_dir("Pick LABELS ROOT (folder containing utility subfolders)")
    lst_dir     = pick_dir("Pick LST FOLDER (contains multiple .lst)")
    out_root    = pick_dir("Pick OUTPUT folder (or create)", mustexist=False)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    # Build image index + sizes once (big speedup)
    build_image_index(Path(images_dir))

    # Parse all LSTs
    poses = parse_all_lsts(Path(lst_dir))

    # Discover utility subfolders
    util_folders = [p for p in Path(labels_root).iterdir() if p.is_dir()]
    util_folders.sort(key=lambda p: p.name.lower())

    summary = []
    for util_folder in util_folders:
        label_dir = find_label_dir_for_utility(util_folder)
        if label_dir is None:
            print(f"Skipping '{util_folder.name}': no .txt labels found")
            summary.append((util_folder.name, 0))
            continue

        util_out = Path(out_root) / util_folder.name
        util_out.mkdir(parents=True, exist_ok=True)

        n = process_one_utility(
            util_name_folder=util_folder.name,
            labels_dir=label_dir,
            poses=poses,
            out_root=util_out,
        )
        summary.append((util_folder.name, n))

    # Summary file
    lines = ["UTILITY,FEATURES_WRITTEN"]
    for nm, n in summary:
        lines.append(f"{nm},{n}")
    (Path(out_root)/"summary.csv").write_text("\n".join(lines), encoding="utf-8")

    done = "\n".join([f"{nm}: {n}" for nm, n in summary])
    messagebox.showinfo("Done", f"Finished all utilities.\n\n{done}\n\nSee summary.csv.")

if __name__ == "__main__":
    main()
