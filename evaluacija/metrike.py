"""Pomoćne funkcije za usporedbu algoritamskih rezultata sa stvarnim
(ground truth) vrijednostima: sparivanje okvira po IoU-u i izračun
postotka punjenja iz ručno označene granice tekućine.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from algoritmi.common import BBox, iou

Match = Tuple[int, int]
# (naziv, indeks najsličnijeg stvarnog okvira ili -1, udio površine unutar njega)
FPClassification = Tuple[str, int, float]

FRAGMENT_CONTAINMENT_THRESHOLD = 0.7
"""Prag udjela površine detektiranog okvira unutar stvarnog (GT) okvira za
prepoznavanje fragmenta: stvaran komad iste boce (npr. samo grlo/rame
odvojeno konturom od tijela) gotovo uvijek leži GOTOVO POSVE unutar njenog
stvarnog okvira (udio blizu 1.0), dok nepovezan lažni pozitiv (predmet koji
nije ta boca) tek rubno dodiruje neki stvaran okvir ili uopće ne. 0.7
ostavlja marginu za nesavršene rubove okvira."""


def match_boxes(
    gt_boxes: Sequence[BBox], det_boxes: Sequence[BBox], iou_threshold: float
) -> Tuple[List[Match], List[int], List[int]]:
    """Pohlepno (greedy) sparivanje po opadajućem IoU-u - dovoljno dobro za
    mali broj okvira po slici (koliko ih realno ima boca na jednoj
    fotografiji), pa puni mađarski algoritam nije potreban.

    Vraća:
    - matches: lista (gt_idx, det_idx) parova čiji je IoU >= praga,
    - unmatched_gt: indeksi stvarnih boca koje nijedan detektirani okvir
      nije pogodio (promašaji / false negatives),
    - unmatched_det: indeksi detektiranih okvira bez odgovarajuće stvarne
      boce (lažni pozitivi / false positives).
    """
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
    """Udio površine okvira `inner` koji leži unutar `outer`. Za razliku od
    `iou`, ne kažnjava kad je `inner` mnogo manji od `outer` - upravo to
    svojstvo treba za prepoznavanje fragmenta (mali okvir gotovo posve
    unutar prevelikog stvarnog okvira boce, koji sam za sebe ima nizak IoU
    pa ga `match_boxes` ispravno ne broji kao pogodak)."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    inner_area = aw * ah
    return inter / inner_area if inner_area else 0.0


def best_containment(det_box: BBox, gt_boxes: Sequence[BBox]) -> Tuple[int, float]:
    """Vraća (indeks, udio) stvarnog okvira unutar kojeg `det_box` najviše
    leži (vidi `_containment`), ili (-1, 0.0) ako nema stvarnih okvira."""
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
    """Razvrstava svaki nespareni detektirani okvir (lažni pozitiv,
    `unmatched_det` iz `match_boxes`) u:

    - "fragmentacija": gotovo posve (>= `FRAGMENT_CONTAINMENT_THRESHOLD`)
      unutar stvarnog okvira koji je VEĆ spario neki DRUGI, bolji okvir -
      suvišan dodatan komad boce koja je već ispravno pronađena (npr.
      `lab_euklidska` na 20260723_181228...: ista boca zna se prelomiti u
      2-3 okvira, samo jedan od njih se broji kao pogodak).
    - "djelomicno": gotovo posve unutar stvarnog okvira koji NIJE spario
      nijedan okvir - ovaj okvir je "blizu" ispravnoj detekciji te
      PROMAŠENE boce, samo nedovoljno velik/dobro pozicioniran da pređe
      `IOU_MATCH_THRESHOLD` (razlikuje "skoro" promašaj od potpunog).
    - "pravi": ne preklapa se značajno ni s jednim stvarnim okvirom -
      stvarno pogrešna detekcija (npr. bocoliki predmet koji nije boca).

    Vraća rječnik {det_indeks: (naziv, najsličniji_gt_indeks, udio)}."""
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
    """Stvarni postotak punjenja iz ručno označene granice: analogno
    formuli koju algoritmi koriste za izračun iz svoje detektirane
    granice (`fill_percent = 100 * (donji rub - granica) / visina`), samo
    primijenjeno na `level_y` koji je odredio čovjek."""
    _x, y, _w, h = bbox
    if h <= 0:
        return 0.0
    return float(np.clip(100.0 * (y + h - level_y) / h, 0.0, 100.0))
