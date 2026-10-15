"""Detekcija boca i razine tekućine pomoću gradijenta boje u CIELAB-u.

Rubovi se računaju iz euklidskih razlika susjednih piksela. Za procjenu
razine koriste se CIELAB komponente i HSV zasićenje. RGB inačica
primjenjuje isti postupak u drugom prostoru boja."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import common

ALGORITHM_NAME = "Euklidska udaljenost - gradijent boje (CIELAB)"
OUTPUT_DIRNAME = "lab_euklidska"
SEARCH_RANGE = (0.15, 0.90)  # Širi raspon čuva visoke i niske razine; uži može pomaknuti slabe procjene prema sredini.
MARGIN_RATIO = 0.18  # Bočna margina kao udio širine okvira.
N_SEGMENTS = 12  # Broj segmenata odabran kalibracijom.


def edge_map(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    gx = np.zeros(lab.shape[:2])
    gy = np.zeros(lab.shape[:2])

    diff_x = lab[:, 2:, :] - lab[:, :-2, :]
    diff_y = lab[2:, :, :] - lab[:-2, :, :]

    gx[:, 1:-1] = np.linalg.norm(diff_x, axis=2)
    gy[1:-1, :] = np.linalg.norm(diff_y, axis=2)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val * 255
    return magnitude.astype(np.uint8)


def _boxes_and_binary(image_bgr: np.ndarray):
    """Vraća okvire i binarnu kartu dobivenu CIELAB gradijentom."""
    return common.localize_gradient(image_bgr, edge_map)


def detect_bottle_regions(image_bgr: np.ndarray) -> List[common.BBox]:
    boxes, _binary = _boxes_and_binary(image_bgr)
    return boxes


def compute_outline_edges(image_bgr: np.ndarray, binary: np.ndarray = None) -> np.ndarray:
    """Dopunjuje binarnu kartu rubovima zasićenja i uklanja regije kože.

    Ako binary nije zadan, lokalizacija se izračunava ovdje."""
    if binary is None:
        _, binary = _boxes_and_binary(image_bgr)
    return common.exclude_skin(
        cv2.bitwise_or(binary, common.saturation_edge_map(image_bgr)), image_bgr
    )


def native_outline(image_bgr: np.ndarray, bbox: common.BBox, outline_edges) -> "np.ndarray | None":
    """Prati obris na karti rubova; vraća None ako nema dovoljno signala."""
    return common.bottle_outline(bbox, outline_edges)


def _feature_map(roi_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    saturation = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float64)
    return np.concatenate([lab, saturation[:, :, None]], axis=2)


def estimate_liquid_level(
    image_bgr: np.ndarray, bbox: common.BBox, outline: "np.ndarray | None" = None
) -> Tuple[List[common.Point], float, float]:
    x, y, w, h = bbox
    roi_bgr = image_bgr[y:y + h, x:x + w]
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    feature_map = _feature_map(roi_bgr)

    search_range = common.body_search_range(outline, bbox, SEARCH_RANGE)
    points, confidence = common.boundary_curve(
        feature_map, gray_roi, MARGIN_RATIO, search_range, N_SEGMENTS
    )
    curve_abs, fill_percent = common.finalize_boundary(points, bbox)
    return curve_abs, fill_percent, confidence


def process_image(path, output_dir) -> Tuple[Path, List[common.BottleResult]]:
    image = common.load_image(path)
    bboxes, binary = _boxes_and_binary(image)
    outline_edges = compute_outline_edges(image, binary)

    results: List[common.BottleResult] = []
    for bbox in bboxes:
        outline = native_outline(image, bbox, outline_edges)
        curve_abs, fill_percent, confidence = estimate_liquid_level(image, bbox, outline)
        fill_percent_volume = common.volumetric_fill_percent(outline, bbox, fill_percent)
        results.append(common.BottleResult(bbox, curve_abs, fill_percent, outline, fill_percent_volume, confidence))

    annotated = common.annotate_bottles(image, results, ALGORITHM_NAME)
    out_path = Path(output_dir) / f"{Path(path).stem}_lab_euklidska.png"
    common.save_image(out_path, annotated)
    return out_path, results
