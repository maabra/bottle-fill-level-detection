"""Vizualna usporedba ALGORITMA s LJUDSKIM ground truthom.

Za svaku označenu sliku iscrtava na istom okviru (ground truth okvir):
  - GT razinu (ljudska istina) kao debelu ZELENU vodoravnu crtu + "GT XX%",
  - razinu svakog algoritma kao tanju obojenu vodoravnu crtu + "alg XX%",
  sve na ISTOM (GT) okviru pa je odstupanje = okomiti razmak između crta.
Time se na prvi pogled vidi tko je blizu čovjeku, a tko bježi - i je li
sam GT ispravno postavljen (crta prati stvarnu površinu tekućine).

Sve slike se spremaju u `evaluacija/usporedba/`, a na ekran se ispiše
tablica GT% vs svaki algoritam% po boci (s pogreškom u postotnim bodovima).

Pokretanje:  python -m evaluacija.usporedba
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algoritmi import common, lab_euklidska, rgb_euklidska, kmeans_euklidska, chanvese_euklidska
from evaluacija import metrike
from evaluacija.oznacavanje import load_ground_truth

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "boce"
OUT_DIR = Path(__file__).resolve().parent / "usporedba"
IOU_MATCH_THRESHOLD = 0.4

# (naziv, modul, BGR boja crte)
ALGOS = [
    ("lab", lab_euklidska, (255, 120, 0)),      # plava
    ("rgb", rgb_euklidska, (255, 0, 200)),      # ljubičasta
    ("kmeans", kmeans_euklidska, (0, 140, 255)),  # narančasta
    ("chanvese", chanvese_euklidska, (0, 200, 255)),  # žuta
]
GT_COLOR = (60, 220, 60)


def _algo_levels(module, image) -> List[tuple]:
    """Vrati listu (bbox, fill_percent) za sve detekcije jednog algoritma."""
    if hasattr(module, "_boxes_and_binary"):
        boxes, binary = module._boxes_and_binary(image)
        edges = module.compute_outline_edges(image, binary)
    else:
        boxes = module.detect_bottle_regions(image)
        edges = module.compute_outline_edges(image) if hasattr(module, "compute_outline_edges") else None
    out = []
    for bbox in boxes:
        outline = module.native_outline(image, bbox, edges)
        _curve, fill, _conf = module.estimate_liquid_level(image, bbox, outline)
        out.append((bbox, fill))
    return out


def _draw_hline_on_gt(img, gt_box, fill_percent, color, label, slot):
    """Nacrtaj vodoravnu crtu na GT okviru na visini koja odgovara
    `fill_percent` (kao da je boca puna toliko %), + oznaku."""
    x, y, w, h = gt_box
    ly = int(y + h * (1.0 - fill_percent / 100.0))
    cv2.line(img, (x, ly), (x + w, ly), color, 2, cv2.LINE_AA)
    common._draw_label(img, label, (x + w + 6, y + 24 + slot * 26), scale=0.6, fg=(255, 255, 255), bg=color)


def compare_image(image, annotations) -> tuple:
    out = image.copy()
    gt_boxes = [tuple(a["bbox"]) for a in annotations]
    gt_fills = [metrike.ground_truth_fill_percent(gt_boxes[i], annotations[i]["level_y"])
                for i in range(len(annotations))]

    # Detektiraj po algoritmu i spari s GT okvirima.
    per_algo_match: Dict[str, Dict[int, float]] = {}
    for name, module, _color in ALGOS:
        dets = _algo_levels(module, image)
        det_boxes = [d[0] for d in dets]
        matches, _ug, _ud = metrike.match_boxes(gt_boxes, det_boxes, IOU_MATCH_THRESHOLD)
        per_algo_match[name] = {gi: dets[di][1] for gi, di in matches}

    rows = []
    for gi, gt_box in enumerate(gt_boxes):
        x, y, w, h = gt_box
        cv2.rectangle(out, (x, y), (x + w, y + h), GT_COLOR, 2, cv2.LINE_AA)
        gtf = gt_fills[gi]
        gly = int(y + h * (1.0 - gtf / 100.0))
        cv2.line(out, (x, gly), (x + w, gly), GT_COLOR, 4, cv2.LINE_AA)
        common._draw_label(out, f"GT {gtf:.0f}%", (x + w + 6, y - 4), scale=0.6, fg=(20, 20, 20), bg=GT_COLOR)

        row = {"bottle": gi, "GT": round(gtf, 1)}
        for slot, (name, _m, color) in enumerate(ALGOS):
            if gi in per_algo_match[name]:
                f = per_algo_match[name][gi]
                _draw_hline_on_gt(out, gt_box, f, color, f"{name} {f:.0f}% ({f - gtf:+.0f})", slot)
                row[name] = round(f, 1)
            else:
                row[name] = None
        rows.append(row)
    return out, rows


def main() -> None:
    gt = load_ground_truth()
    if not gt:
        print("Nema ground truth podataka. Prvo: python -m evaluacija.oznacavanje")
        return
    common.ensure_dir(OUT_DIR)

    all_rows = []
    n = 0
    for name, annotations in gt.items():
        path = INPUT_DIR / name
        if not path.exists() or not annotations:
            continue
        image = common.load_image(path)
        out, rows = compare_image(image, annotations)
        common.save_image(OUT_DIR / f"{Path(name).stem}_usporedba.png", out)
        for r in rows:
            r["slika"] = name[:24]
            all_rows.append(r)
        n += 1

    # Tablica na ekran.
    print(f"\nUsporedba GT vs algoritmi ({n} slika) - pogreška u p.b. iza svakog %:\n")
    hdr = f"{'slika':<26}{'#':>2} {'GT':>6} " + " ".join(f"{a[0]:>8}" for a in ALGOS)
    print(hdr); print("-" * len(hdr))
    for r in all_rows:
        line = f"{r['slika']:<26}{r['bottle']:>2} {r['GT']:>6} "
        for a in ALGOS:
            v = r[a[0]]
            line += f"{(f'{v:.0f}({v-r[\"GT\"]:+.0f})' if v is not None else '-'):>8} "
        print(line)
    print(f"\nSlike spremljene u: {OUT_DIR}")


if __name__ == "__main__":
    main()
