"""Evaluacija PO KATEGORIJAMA boca/scena - za diplomski rad.

Imena datoteka u `boce/` kodiraju četiri kategorije (izračunate, ne
procijenjene okom - vidi projektne bilješke):
    {izvorno_ime}_{jedna|vise}_{slabo|umjereno|jako}_{niska|visoka}_{jednostavna|kompleksna}.webp
    (broj boca)   (svjetlina)                (punjenje)      (složenost pozadine)

Za svaki algoritam i svaku vrijednost svake kategorije računa:
    n_boca (GT), odziv, preciznost, prosj. IoU (lokalizacija) i MAE razine
    (postotni bodovi, samo na uspješno sparenim bocama).

Time se vidi GDJE svaka od tri paradigme radi, a gdje pada - npr. prozirne
boce na složenoj (kompleksnoj) pozadini su granica svih metoda, dok su
jednostavne studijske scene ~riješene. Rezultat: `evaluacija/rezultati/
po_kategorijama.csv` + čitljive tablice na ekranu.

Pokretanje:  python -m evaluacija.evaluacija_po_kategorijama
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from algoritmi import common, lab_euklidska, rgb_euklidska, kmeans_euklidska, chanvese_euklidska
from evaluacija import metrike

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "boce"
RESULTS_DIR = ROOT / "evaluacija" / "rezultati"
IOU_MATCH = 0.4

ALGORITHMS = {
    "lab": lab_euklidska,
    "rgb": rgb_euklidska,
    "kmeans": kmeans_euklidska,
    "chanvese": chanvese_euklidska,
}

# Redoslijed vrijednosti unutar svake dimenzije (za uredne tablice)
DIMENSIONS = {
    "broj":       ("jedna", "vise"),
    "svjetlina":  ("slabo", "umjereno", "jako"),
    "punjenje":   ("niska", "visoka"),
    "slozenost":  ("jednostavna", "kompleksna"),
}


def parse_categories(name: str) -> Dict[str, str]:
    """Zadnja četiri tokena imena (bez .webp) su kategorije."""
    stem = name.rsplit(".", 1)[0]
    toks = stem.split("_")
    broj, svjetlina, punjenje, slozenost = toks[-4], toks[-3], toks[-2], toks[-1]
    return {"broj": broj, "svjetlina": svjetlina, "punjenje": punjenje, "slozenost": slozenost}


def _outline_edges(mod, image):
    fn = getattr(mod, "compute_outline_edges", None)
    if fn is None:
        return None
    try:
        return fn(image)
    except TypeError:
        return fn(image, None)


class Acc:
    """Akumulator statistike jedne (algoritam, kategorija) grupe."""
    __slots__ = ("n_gt", "tp", "fp", "ious", "abserr")

    def __init__(self):
        self.n_gt = 0
        self.tp = 0
        self.fp = 0
        self.ious: List[float] = []
        self.abserr: List[float] = []

    def rec(self):
        return self.tp / self.n_gt if self.n_gt else float("nan")

    def prec(self):
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    def iou(self):
        return float(np.mean(self.ious)) if self.ious else float("nan")

    def mae(self):
        return float(np.mean(self.abserr)) if self.abserr else float("nan")


def evaluate():
    gt = __import__("json").loads((ROOT / "evaluacija" / "ground_truth.json").read_text(encoding="utf-8"))
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() == ".webp" and p.name in gt)

    # acc[alg][dim][value] = Acc ; plus overall acc[alg]["_ukupno"]["_"]
    acc: Dict[str, Dict[str, Dict[str, Acc]]] = {
        a: defaultdict(lambda: defaultdict(Acc)) for a in ALGORITHMS
    }

    for path in images:
        image = common.load_image(path)
        gts = [tuple(b["bbox"]) for b in gt[path.name]]
        gt_fills = [metrike.ground_truth_fill_percent(tuple(b["bbox"]), b["level_y"]) for b in gt[path.name]]
        cats = parse_categories(path.name)
        # grupe kojima ova slika/boce pripadaju (po dimenzijama + ukupno)
        groups = [("_ukupno", "_")] + [(dim, cats[dim]) for dim in DIMENSIONS]

        for akey, mod in ALGORITHMS.items():
            det = mod.detect_bottle_regions(image)
            matches, unmatched_gt, unmatched_det = metrike.match_boxes(gts, det, IOU_MATCH)
            oe = _outline_edges(mod, image)
            # razina po sparenoj boci
            errs = {}
            for gi, di in matches:
                outline = mod.native_outline(image, det[di], oe)
                _c, fill, _conf = mod.estimate_liquid_level(image, det[di], outline)
                errs[gi] = abs(fill - gt_fills[gi])
            for dim, val in groups:
                a = acc[akey][dim][val]
                a.n_gt += len(gts)
                a.tp += len(matches)
                a.fp += len(unmatched_det)
                for gi, di in matches:
                    a.ious.append(common.iou(gts[gi], det[di]))
                    a.abserr.append(errs[gi])

    return acc


def print_and_save(acc):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def line(dim, val):
        cells = []
        for akey in ALGORITHMS:
            a = acc[akey][dim][val]
            cells.append((akey, a))
            rows.append({
                "dimenzija": dim, "kategorija": val, "algoritam": akey,
                "n_boca": a.n_gt, "TP": a.tp, "FP": a.fp,
                "odziv": round(a.rec(), 3), "preciznost": round(a.prec(), 3),
                "prosj_IoU": round(a.iou(), 3), "MAE_%": round(a.mae(), 2),
            })
        return cells

    def show(title, dim, values):
        print(f"\n=== {title} ===")
        print(f"{'kategorija':13s} {'alg':9s} {'n':>3} {'odziv':>6} {'prec':>6} {'IoU':>6} {'MAE%':>6}")
        for val in values:
            for akey, a in line(dim, val):
                print(f"{val:13s} {akey:9s} {a.n_gt:3d} {a.rec():6.3f} {a.prec():6.3f} {a.iou():6.3f} {a.mae():6.2f}")
            print()

    show("UKUPNO (svih 65 boca)", "_ukupno", ["_"])
    show("Po BROJU boca", "broj", DIMENSIONS["broj"])
    show("Po SVJETLINI", "svjetlina", DIMENSIONS["svjetlina"])
    show("Po PUNJENJU", "punjenje", DIMENSIONS["punjenje"])
    show("Po SLOŽENOSTI pozadine", "slozenost", DIMENSIONS["slozenost"])

    out = RESULTS_DIR / "po_kategorijama.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSpremljeno: {out}")


if __name__ == "__main__":
    print_and_save(evaluate())
