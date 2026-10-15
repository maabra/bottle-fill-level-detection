"""Sparivanje detektiranih i referentnih okvira te izračun stvarne
razine punjenja iz ručnih oznaka."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from algoritmi.common import BBox, iou

Match = Tuple[int, int]
# (naziv, indeks najsličnijeg stvarnog okvira ili -1, udio površine unutar njega)
FPClassification = Tuple[str, int, float]

FRAGMENT_CONTAINMENT_THRESHOLD = 0.7
"""Najmanji udio detekcije unutar stvarnog okvira za oznaku fragmenta.
Prag ostavlja prostor za manje pogreške na rubovima okvira."""


def match_boxes(
    gt_boxes: Sequence[BBox], det_boxes: Sequence[BBox], iou_threshold: float
) -> Tuple[List[Match], List[int], List[int]]:
    """Pohlepno sparuje okvire po opadajućem IoU-u.

    Vraća parove (gt_idx, det_idx) iznad praga, indekse promašenih
    referentnih boca i indekse nesparenih detekcija."""
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for di, det in enumerate(det_boxes):
            score = iou(gt, det)
            if score >= iou_threshold:
                candidates.append((score, gi, di))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt, matched_det = set(), set()
    matches: List[Match] = []
    for _score, gi, di in candidates:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        matches.append((gi, di))

    unmatched_gt = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    unmatched_det = [di for di in range(len(det_boxes)) if di not in matched_det]
    return matches, unmatched_gt, unmatched_det


def _containment(inner: BBox, outer: BBox) -> float:
    """Vraća udio površine inner sadržan u outer, neovisno o veličini outer."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    inner_area = aw * ah
    return inter / inner_area if inner_area else 0.0


def best_containment(det_box: BBox, gt_boxes: Sequence[BBox]) -> Tuple[int, float]:
    """Vraća indeks referentnog okvira s najvećom sadržanošću i taj udio.

    Bez preklapanja vraća (-1, 0.0)."""
    best_gi, best_score = -1, 0.0
    for gi, gt in enumerate(gt_boxes):
        score = _containment(det_box, gt)
        if score > best_score:
            best_gi, best_score = gi, score
    return best_gi, best_score


def classify_false_positives(
    gt_boxes: Sequence[BBox],
    det_boxes: Sequence[BBox],
    unmatched_det: Sequence[int],
    matched_gt: Sequence[int],
) -> Dict[int, FPClassification]:
    """Razvrstava nesparene detekcije prema sadržanosti u stvarnim okvirima.

    fragmentacija: dio boce koju je drugi okvir već uspješno pogodio.
    djelomicno: dio promašene boce, nedovoljan za IoU prag.
    pravi: detekcija bez dovoljne sadržanosti u ijednom stvarnom okviru.
    Vraća {det_indeks: (naziv, gt_indeks, udio)}."""
    matched_gt_set = set(matched_gt)
    labels: Dict[int, FPClassification] = {}
    for di in unmatched_det:
        best_gi, best_score = best_containment(det_boxes[di], gt_boxes)
        if best_score >= FRAGMENT_CONTAINMENT_THRESHOLD:
            label = "fragmentacija" if best_gi in matched_gt_set else "djelomicno"
        else:
            label = "pravi"
        labels[di] = (label, best_gi, best_score)
    return labels


def ground_truth_fill_percent(bbox: BBox, level_y: float) -> float:
    """Računa udio punjenja iz ručne granice: 100 * (y + h - level_y) / h."""
    _x, y, _w, h = bbox
    if h <= 0:
        return 0.0
    return float(np.clip(100.0 * (y + h - level_y) / h, 0.0, 100.0))
