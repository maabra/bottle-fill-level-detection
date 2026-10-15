"""Vrednuje četiri algoritma prema ručno označenim bocama.

Svaki algoritam lokalizira boce i procjenjuje razinu, a okviri se
sparuju po IoU-u. Lokalizacija se mjeri preciznošću, odzivom i IoU-om;
pogreška razine računa se samo za sparene boce, u postotnim bodovima.
Mjeri se i vrijeme obrade. Označene izlazne slike ovdje se ne spremaju.

CSV izvještaji u evaluacija/rezultati prikazuju zadnje pokretanje.
Datoteka povijest.csv čuva prethodne rezultate i dodaje vremensku oznaku.
Pokretanje: python -m evaluacija.pokreni_evaluaciju"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    # Dodaj korijen projekta za izravno pokretanje skripte iz IDE-a.
    sys.path.insert(0, str(_PROJECT_ROOT))

from algoritmi import common, rgb_euklidska, lab_euklidska, kmeans_euklidska, chanvese_euklidska
from evaluacija import metrike
from evaluacija.oznacavanje import GROUND_TRUTH_PATH, load_ground_truth

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "boce"
RESULTS_DIR = Path(__file__).resolve().parent / "rezultati"

# Tri paradigme (gradijent/CIELAB, K-means, Chan-Vese) + RGB gradijent
# kao ablacijska usporedba prostora boja unutar gradijentne metode.
ALGORITHMS = {
    "lab_euklidska": lab_euklidska,
    "rgb_euklidska": rgb_euklidska,
    "kmeans_euklidska": kmeans_euklidska,
    "chanvese_euklidska": chanvese_euklidska,
}

IOU_MATCH_THRESHOLD = 0.4


def evaluate_algorithm(module, ground_truth: Dict[str, List[dict]], input_dir: Path):
    detail_rows: List[dict] = []
    ious: List[float] = []
    fill_errors: List[float] = []
    confident_errors: List[float] = []
    unconfident_errors: List[float] = []
    tp = fn = fp = 0
    fp_fragmentacija = fp_djelomicno = fp_pravi = 0
    fn_s_djelomicnom = 0
    total_time = 0.0
    n_images = 0
    # (slika, gt_indeks) povezuje procjene iste boce za ansambl.
    gt_estimates: Dict[tuple, tuple] = {}

    for image_name, annotations in ground_truth.items():
        if not annotations:
            continue
        path = input_dir / image_name
        if not path.exists():
            print(f"  upozorenje: {image_name} nedostaje u {input_dir}, preskačem")
            continue

        image = common.load_image(path)
        n_images += 1

        t0 = time.perf_counter()
        # Ponovno upotrijebi binarnu kartu RGB/CIELAB lokalizacije za obris.
        # Svaki algoritam računa svoj obris kako bi evaluacija koristila
        # isti raspon pretrage kao process_image.
        if hasattr(module, "_boxes_and_binary"):
            detected_boxes, binary = module._boxes_and_binary(image)
            outline_edges = module.compute_outline_edges(image, binary)
        else:
            detected_boxes = module.detect_bottle_regions(image)
            # K-means koristi granice klastera; Chan-Vese izravno segmentira siluetu.
            outline_edges = (
                module.compute_outline_edges(image)
                if hasattr(module, "compute_outline_edges") else None
            )
        detected = []
        for bbox in detected_boxes:
            outline = module.native_outline(image, bbox, outline_edges)
            _curve, det_fill, confidence = module.estimate_liquid_level(image, bbox, outline)
            detected.append((bbox, det_fill, confidence))
        total_time += time.perf_counter() - t0

        gt_boxes = [tuple(a["bbox"]) for a in annotations]
        det_boxes = [d[0] for d in detected]
        matches, unmatched_gt, unmatched_det = metrike.match_boxes(
            gt_boxes, det_boxes, IOU_MATCH_THRESHOLD
        )
        # Odvoji fragmente već pronađenih boca, djelomične detekcije i ostale FP.
        fp_classification = metrike.classify_false_positives(
            gt_boxes, det_boxes, unmatched_det, [gi for gi, _di in matches]
        )
        near_miss_gt = {gi for label, gi, _score in fp_classification.values() if label == "djelomicno"}

        for gi, di in matches:
            gt_box = gt_boxes[gi]
            gt_fill = metrike.ground_truth_fill_percent(gt_box, annotations[gi]["level_y"])
            det_box, det_fill, confidence = detected[di]
            score = metrike.iou(gt_box, det_box)
            err = det_fill - gt_fill
            is_confident = confidence >= common.CONFIDENCE_THRESHOLD
            tp += 1
            ious.append(score)
            fill_errors.append(err)
            (confident_errors if is_confident else unconfident_errors).append(err)
            gt_estimates[(image_name, gi)] = (det_fill, confidence)
            detail_rows.append({
                "algoritam": module.OUTPUT_DIRNAME, "slika": image_name, "status": "OK",
                "iou": round(score, 3), "stvarni_%": round(gt_fill, 1),
                "izmjereni_%": round(det_fill, 1), "pogreska_%": round(err, 1),
                "pouzdanost": round(confidence, 2), "pouzdano": "da" if is_confident else "ne",
                "fp_tip": "",
            })

        for gi in unmatched_gt:
            fn += 1
            if gi in near_miss_gt:
                fn_s_djelomicnom += 1
            gt_box = gt_boxes[gi]
            gt_fill = metrike.ground_truth_fill_percent(gt_box, annotations[gi]["level_y"])
            detail_rows.append({
                "algoritam": module.OUTPUT_DIRNAME, "slika": image_name,
                "status": "PROMASENO_DJELOMICNO" if gi in near_miss_gt else "PROMASENO",
                "iou": "", "stvarni_%": round(gt_fill, 1), "izmjereni_%": "", "pogreska_%": "",
                "pouzdanost": "", "pouzdano": "", "fp_tip": "",
            })

        for di in unmatched_det:
            fp += 1
            fp_tip, _best_gi, _best_score = fp_classification[di]
            if fp_tip == "fragmentacija":
                fp_fragmentacija += 1
            elif fp_tip == "djelomicno":
                fp_djelomicno += 1
            else:
                fp_pravi += 1
            _det_box, det_fill, confidence = detected[di]
            detail_rows.append({
                "algoritam": module.OUTPUT_DIRNAME, "slika": image_name, "status": "LAZNI_POZITIV",
                "iou": "", "stvarni_%": "", "izmjereni_%": round(det_fill, 1), "pogreska_%": "",
                "pouzdanost": round(confidence, 2), "pouzdano": "da" if confidence >= common.CONFIDENCE_THRESHOLD else "ne",
                "fp_tip": fp_tip,
            })

    errors = np.array(fill_errors, dtype=np.float64)
    conf_errors = np.array(confident_errors, dtype=np.float64)
    unconf_errors = np.array(unconfident_errors, dtype=np.float64)
    summary = {
        "algoritam": module.OUTPUT_DIRNAME,
        "n_slika": n_images,
        "n_boca_gt": tp + fn,
        "TP": tp, "FN": fn, "FP": fp,
        "preciznost": tp / (tp + fp) if (tp + fp) else float("nan"),
        "odziv": tp / (tp + fn) if (tp + fn) else float("nan"),
        "prosj_IoU": float(np.mean(ious)) if len(ious) else float("nan"),
        "MAE_%": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
        "RMSE_%": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
        "max_pogreska_%": float(np.max(np.abs(errors))) if len(errors) else float("nan"),
        "prosj_vrijeme_s": total_time / n_images if n_images else float("nan"),
    }
    # Pouzdanost se izvještava zasebno kako stupci povijesnog CSV-a ostaju isti.
    confidence_summary = {
        "algoritam": module.OUTPUT_DIRNAME,
        "n_pouzdano": len(confident_errors),
        "n_nepouzdano": len(unconfident_errors),
        "MAE_pouzdano_%": float(np.mean(np.abs(conf_errors))) if len(conf_errors) else float("nan"),
        "RMSE_pouzdano_%": float(np.sqrt(np.mean(conf_errors ** 2))) if len(conf_errors) else float("nan"),
        "MAE_nepouzdano_%": float(np.mean(np.abs(unconf_errors))) if len(unconf_errors) else float("nan"),
        "RMSE_nepouzdano_%": float(np.sqrt(np.mean(unconf_errors ** 2))) if len(unconf_errors) else float("nan"),
    }
    # Razloži FP na fragmente, djelomične detekcije i ostale pogrešne okvire.
    # Zaseban izvještaj čuva postojeći format povijesti.
    fragment_summary = {
        "algoritam": module.OUTPUT_DIRNAME,
        "FP_fragmentacija": fp_fragmentacija,
        "FP_djelomicno": fp_djelomicno,
        "FP_pravi": fp_pravi,
        "FN_s_djelomicnom_det": fn_s_djelomicnom,
        "FN_potpuno_promaseno": fn - fn_s_djelomicnom,
    }
    return summary, detail_rows, confidence_summary, gt_estimates, fragment_summary


def compute_ensemble(
    ground_truth: Dict[str, List[dict]], all_gt_estimates: Dict[str, Dict[tuple, tuple]]
):
    """Spaja procjene medijanom za svaku istu referentnu bocu.

    Prednost imaju procjene iznad praga pouzdanosti. Ako takvih nema,
    koriste se sve dostupne procjene, a boca se broji i u
    n_bez_pouzdane_procjene. Zajedničke pogreške svih metoda medijan
    ne može ukloniti."""
    gt_fill_lookup: Dict[tuple, float] = {}
    for image_name, annotations in ground_truth.items():
        for gi, a in enumerate(annotations):
            gt_fill_lookup[(image_name, gi)] = metrike.ground_truth_fill_percent(
                tuple(a["bbox"]), a["level_y"]
            )

    all_keys = set()
    for estimates in all_gt_estimates.values():
        all_keys.update(estimates.keys())

    errors: List[float] = []
    rows: List[dict] = []
    n_no_confident = 0
    for key in sorted(all_keys, key=lambda k: (k[0], k[1])):
        per_algo = [
            (algo, estimates[key][0], estimates[key][1])
            for algo, estimates in all_gt_estimates.items() if key in estimates
        ]
        confident = [fill for _, fill, conf in per_algo if conf >= common.CONFIDENCE_THRESHOLD]
        chosen = confident if confident else [fill for _, fill, _ in per_algo]
        if not confident:
            n_no_confident += 1
        median_fill = float(np.median(chosen))
        gt_fill = gt_fill_lookup[key]
        err = median_fill - gt_fill
        errors.append(err)
        rows.append({
            "slika": key[0], "gt_indeks": key[1],
            "n_doprinosa": len(per_algo), "n_pouzdanih": len(confident),
            "stvarni_%": round(gt_fill, 1), "medijan_%": round(median_fill, 1),
            "pogreska_%": round(err, 1),
        })

    errors_arr = np.array(errors, dtype=np.float64)
    summary = {
        "algoritam": "ansambl (medijan)",
        "n_boca": len(errors),
        "n_bez_pouzdane_procjene": n_no_confident,
        "MAE_%": float(np.mean(np.abs(errors_arr))) if len(errors_arr) else float("nan"),
        "RMSE_%": float(np.sqrt(np.mean(errors_arr ** 2))) if len(errors_arr) else float("nan"),
        "max_pogreska_%": float(np.max(np.abs(errors_arr))) if len(errors_arr) else float("nan"),
    }
    return summary, rows


def _fmt(value) -> str:
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def print_summary_table(summaries: List[dict]) -> None:
    columns = list(summaries[0].keys())
    widths = {c: max(len(c), *(len(_fmt(s[c])) for s in summaries)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for s in summaries:
        print("  ".join(_fmt(s[c]).ljust(widths[c]) for c in columns))


def write_csv(path: Path, rows: List[dict]) -> None:
    common.ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_history(path: Path, rows: List[dict]) -> None:
    """Dodaje rezultate s vremenskom oznakom na kraj povijesnog CSV-a."""
    common.ensure_dir(path.parent)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_with_ts = [{"vrijeme": timestamp, **row} for row in rows]
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_with_ts[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows_with_ts)


def main() -> None:
    ground_truth = load_ground_truth()
    if not ground_truth:
        print(
            f"Nema ground truth podataka ({GROUND_TRUTH_PATH}).\n"
            "Prvo pokreni: python -m evaluacija.oznacavanje"
        )
        return

    n_boxes = sum(len(v) for v in ground_truth.values())
    print(f"Ground truth: {len(ground_truth)} slika, {n_boxes} označenih boca.\n")

    summaries: List[dict] = []
    all_details: List[dict] = []
    confidence_summaries: List[dict] = []
    fragment_summaries: List[dict] = []
    all_gt_estimates: Dict[str, Dict[tuple, tuple]] = {}
    for module in ALGORITHMS.values():
        print(f"--- {module.ALGORITHM_NAME} ---")
        summary, details, confidence_summary, gt_estimates, fragment_summary = evaluate_algorithm(
            module, ground_truth, INPUT_DIR
        )
        summaries.append(summary)
        all_details.extend(details)
        confidence_summaries.append(confidence_summary)
        fragment_summaries.append(fragment_summary)
        all_gt_estimates[module.OUTPUT_DIRNAME] = gt_estimates
        print(
            f"  TP={summary['TP']} FN={summary['FN']} FP={summary['FP']}  "
            f"MAE={_fmt(summary['MAE_%'])}%  RMSE={_fmt(summary['RMSE_%'])}%\n"
        )

    print("=== Sažetak (svi algoritmi) ===")
    print_summary_table(summaries)

    print("\n=== Pouzdanost (P4) - MAE odvojeno za pouzdane/nepouzdane procjene ===")
    print_summary_table(confidence_summaries)

    ensemble_summary, ensemble_rows = compute_ensemble(ground_truth, all_gt_estimates)
    print("\n=== Ansambl (P5) - medijan preko svih algoritama po stvarnoj boci ===")
    print_summary_table([ensemble_summary])

    print(
        "\n=== Razlaganje FP/FN (P6) - fragmentacija vs. stvarni lažni pozitivi, "
        "djelomične vs. potpune promašaje ==="
    )
    print_summary_table(fragment_summaries)

    write_csv(RESULTS_DIR / "sazetak.csv", summaries)
    write_csv(RESULTS_DIR / "detaljno.csv", all_details)
    write_csv(RESULTS_DIR / "pouzdanost.csv", confidence_summaries)
    write_csv(RESULTS_DIR / "fragmentacija.csv", fragment_summaries)
    if ensemble_rows:
        write_csv(RESULTS_DIR / "ansambl.csv", ensemble_rows)
    append_history(RESULTS_DIR / "povijest.csv", summaries)
    print(f"\nSpremljeno: {RESULTS_DIR / 'sazetak.csv'} (samo zadnje pokretanje)")
    print(f"Spremljeno: {RESULTS_DIR / 'detaljno.csv'} (samo zadnje pokretanje, uklj. pouzdanost po boci)")
    print(f"Spremljeno: {RESULTS_DIR / 'pouzdanost.csv'} (samo zadnje pokretanje)")
    print(f"Spremljeno: {RESULTS_DIR / 'fragmentacija.csv'} (samo zadnje pokretanje)")
    print(f"Spremljeno: {RESULTS_DIR / 'ansambl.csv'} (samo zadnje pokretanje, jedan redak po stvarnoj boci)")
    print(f"Dodano u: {RESULTS_DIR / 'povijest.csv'} (raste iz pokretanja u pokretanje)")


if __name__ == "__main__":
    main()
