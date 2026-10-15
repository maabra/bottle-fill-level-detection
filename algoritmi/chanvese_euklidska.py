"""Detekcija boca i razine tekućine Chan-Vese segmentacijom.

Globalna segmentacija izdvaja kandidate, a lokalna dorada određuje
siluetu boce. Procjena razine koristi srednje vrijednosti boje i
zasićenja unutar siluete, bez gradijentnog pojačanja i maske etikete."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import common

ALGORITHM_NAME = "Euklidska udaljenost - Chan-Vese aktivna kontura"
OUTPUT_DIRNAME = "chanvese_euklidska"
SEARCH_RANGE = (0.15, 0.90)  # Širi raspon čuva visoke i niske razine; uži može pomaknuti slabe procjene prema sredini.
MARGIN_RATIO = 0.18  # Bočna margina kao udio širine okvira.
N_SEGMENTS = 12  # Broj segmenata u pretrazi granice.


def detect_bottle_regions(image_bgr: np.ndarray) -> List[common.BBox]:
    """Izdvaja okvire globalnom Chan-Vese segmentacijom i lokalnom doradom."""
    return common.localize_variational(image_bgr)


def native_outline(image_bgr: np.ndarray, bbox: common.BBox, outline_edges=None) -> "np.ndarray | None":
    """Vraća Chan-Vese konturu ili None ako segmentacija ne uspije."""
    return common.chan_vese_outline(image_bgr, bbox)


def estimate_liquid_level(
    image_bgr: np.ndarray, bbox: common.BBox, outline: "np.ndarray | None" = None
) -> Tuple[List[common.Point], float, float]:
    """Procjenjuje razinu iz CIELAB i S značajki unutar Chan-Vese siluete.

    Bez siluete koristi unutrašnjost okvira s bočnim marginama.
    Pretraga po segmentima isključuje gradijentne dodatke i uklapa
    vodoravnu granicu."""
    x, y, w, h = bbox
    roi = image_bgr[y:y + h, x:x + w]
    # CIELAB i zasićenje daju značajke za usporedbu sredina regija.
    feats = common.lab_saturation_features(roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    silhouette = common.chan_vese_silhouette(image_bgr, bbox)
    if silhouette is None or silhouette.sum() < 0.05 * w * h:
        # Bez siluete koristi se unutrašnjost okvira s bočnim marginama.
        silhouette = np.zeros((h, w), dtype=np.uint8)
        mx = int(w * MARGIN_RATIO)
        silhouette[:, mx:max(mx + 1, w - mx)] = 1
    inside = silhouette > 0

    search_range = common.body_search_range(outline, bbox, SEARCH_RANGE)
    # Procjena koristi regijske značajke i vodoravnu granicu, bez gradijentnih dodataka.
    points, confidence = common.boundary_curve(
        feats, gray_roi, MARGIN_RATIO, search_range, N_SEGMENTS, mask=inside,
        use_edge_boost=False, use_label_band=False, curve_degree=0,
    )
    curve_abs, fill_percent = common.finalize_boundary(points, bbox)
    return curve_abs, fill_percent, confidence


def process_image(path, output_dir) -> Tuple[Path, List[common.BottleResult]]:
    image = common.load_image(path)
    bboxes = detect_bottle_regions(image)

    results: List[common.BottleResult] = []
    for bbox in bboxes:
        outline = native_outline(image, bbox)
        curve_abs, fill_percent, confidence = estimate_liquid_level(image, bbox, outline)
        fill_percent_volume = common.volumetric_fill_percent(outline, bbox, fill_percent)
        results.append(common.BottleResult(bbox, curve_abs, fill_percent, outline, fill_percent_volume, confidence))

    annotated = common.annotate_bottles(image, results, ALGORITHM_NAME)
    out_path = Path(output_dir) / f"{Path(path).stem}_chanvese_euklidska.png"
    common.save_image(out_path, annotated)
    return out_path, results
