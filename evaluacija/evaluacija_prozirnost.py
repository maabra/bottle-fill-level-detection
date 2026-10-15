"""Evaluacija PROZIRNE vs OBOJENE boce, ukršteno sa složenošću scene - za rad.

Prozirnost NIJE u imenu datoteke pa se procjenjuje iz slike (proxy): boca je
PROZIRNA ako je unutar okvira nisko zasićenje (S < S_THR) I boja unutrašnjosti
je bliska boji neposredne pozadine (bgDist < DIST_THR u CIELAB-u) - tj. boca
se "stapa" s pozadinom, nema vlastitog distinktnog bojnog signala. Inače
OBOJENA (obojena tekućina, staklo, tamna/neprozirna).

VAŽAN NALAZ (vidi projektne bilješke): prozirnost je konfaundirana s pozadinom
- prozirna boca na BIJELOJ studijskoj pozadini ima nizak bgDist ali je LAKA;
teška je tek prozirna boca na SLOŽENOJ (zakrčenoj) pozadini. Zato se izvještava
UKRŠTENO: (prozirna/obojena) × (jednostavna/složena). Zaključak iz podataka:
dominira SLOŽENOST, ne prozirnost.

Sprema:
  rezultati/prozirnost_po_boci.csv   - klasifikacija svake boce + je li ju
                                        svaki algoritam pronašao (za provjeru/
                                        ručnu doradu klasifikacije)
  rezultati/prozirnost_krizno.csv    - 2×2 metrike (odziv/IoU/MAE) po algoritmu

Pokretanje:  python -m evaluacija.evaluacija_prozirnost
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from algoritmi import common, lab_euklidska, rgb_euklidska, kmeans_euklidska, chanvese_euklidska
from evaluacija import metrike

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "boce"
RESULTS_DIR = ROOT / "evaluacija" / "rezultati"
IOU_MATCH = 0.4
S_THR, DIST_THR = 55.0, 25.0   # pragovi proxy-klasifikacije prozirnosti (vidi docstring)

ALGORITHMS = {"lab": lab_euklidska, "rgb": rgb_euklidska,
              "kmeans": kmeans_euklidska, "chanvese": chanvese_euklidska}


def classify(image, bbox):
    """(klasa, zasićenje, bgDist) - vidi docstring modula."""
    x, y, w, h = bbox
    H, W = image.shape[:2]
    sat = float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[y:y + h, x:x + w, 1].mean())
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    inside = lab[y:y + h, x:x + w].reshape(-1, 3).mean(0)
    pad, bg = 20, []
    if x - pad >= 0:
        bg.append(lab[y:y + h, max(0, x - pad):x].reshape(-1, 3))
    if x + w + pad <= W:
        bg.append(lab[y:y + h, x + w:x + w + pad].reshape(-1, 3))
    dist = float(np.linalg.norm(inside - np.concatenate(bg, 0).mean(0))) if bg else 999.0
    cls = "prozirna" if (sat < S_THR and dist < DIST_THR) else "obojena"
    return cls, sat, dist


def _outline_edges(mod, image):
    fn = getattr(mod, "compute_outline_edges", None)
    if fn is None:
        return None
    try:
        return fn(image)
    except TypeError:
        return fn(image, None)


def run():
    gt = json.loads((ROOT / "evaluacija" / "ground_truth.json").read_text(encoding="utf-8"))
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() == ".webp" and p.name in gt)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_bottle_rows = []
    # cross[alg][cls][komp] = [n, tp, [iou], [abserr]]
    cross = {a: defaultdict(lambda: defaultdict(lambda: [0, 0, [], []])) for a in ALGORITHMS}

    for path in images:
        image = common.load_image(path)
        boxes = gt[path.name]
        gts = [tuple(b["bbox"]) for b in boxes]
        fills = [metrike.ground_truth_fill_percent(tuple(b["bbox"]), b["level_y"]) for b in boxes]
        classes = [classify(image, b["bbox"]) for b in boxes]
        comp = "slozena" if "kompleksna" in path.name else "jednostavna"

        det_by_alg = {}
        for akey, mod in ALGORITHMS.items():
            det = mod.detect_bottle_regions(image)
            matches, _ug, _ud = metrike.match_boxes(gts, det, IOU_MATCH)
            oe = _outline_edges(mod, image)
            matched = {}
            for gi, di in matches:
                outline = mod.native_outline(image, det[di], oe)
                _c, fill, _cf = mod.estimate_liquid_level(image, det[di], outline)
                matched[gi] = (di, common.iou(gts[gi], det[di]), abs(fill - fills[gi]))
            det_by_alg[akey] = matched
            for gi in range(len(gts)):
                cls = classes[gi][0]
                cell = cross[akey][cls][comp]
                cell[0] += 1
                if gi in matched:
                    cell[1] += 1
                    cell[2].append(matched[gi][1])
                    cell[3].append(matched[gi][2])

        for gi, b in enumerate(boxes):
            cls, sat, dist = classes[gi]
            row = {"slika": path.name, "gt_indeks": gi, "klasa": cls,
                   "slozenost": comp, "zasicenje": round(sat, 1), "bgDist": round(dist, 1)}
            for akey in ALGORITHMS:
                row[f"nasao_{akey}"] = int(gi in det_by_alg[akey])
            per_bottle_rows.append(row)

    # --- spremi po-boci ---
    with open(RESULTS_DIR / "prozirnost_po_boci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bottle_rows[0].keys()))
        w.writeheader()
        w.writerows(per_bottle_rows)

    # --- spremi križno + ispis ---
    cross_rows = []
    print(f"{'alg':9s} {'klasa':9s} {'slozenost':12s} {'n':>3} {'odziv':>6} {'IoU':>6} {'MAE%':>6}")
    for akey in ALGORITHMS:
        for cls in ("prozirna", "obojena"):
            for comp in ("jednostavna", "slozena"):
                n, tp, ious, errs = cross[akey][cls][comp]
                rec = tp / n if n else float("nan")
                iou = float(np.mean(ious)) if ious else float("nan")
                mae = float(np.mean(errs)) if errs else float("nan")
                cross_rows.append({"algoritam": akey, "klasa": cls, "slozenost": comp,
                                   "n_boca": n, "TP": tp, "odziv": round(rec, 3),
                                   "prosj_IoU": round(iou, 3), "MAE_%": round(mae, 2)})
                print(f"{akey:9s} {cls:9s} {comp:12s} {n:3d} {rec:6.3f} {iou:6.3f} {mae:6.2f}")
    with open(RESULTS_DIR / "prozirnost_krizno.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
        w.writeheader()
        w.writerows(cross_rows)

    from collections import Counter
    cc = Counter((r["klasa"], r["slozenost"]) for r in per_bottle_rows)
    print("\nBroj boca po ćeliji:", dict(cc))
    print(f"Spremljeno: {RESULTS_DIR / 'prozirnost_po_boci.csv'}")
    print(f"Spremljeno: {RESULTS_DIR / 'prozirnost_krizno.csv'}")


if __name__ == "__main__":
    run()
