"""Pretražuje kombinacije parametara lokalizacije i granice tekućine.

Cilj localization optimizira prosječni F1, a boundary prosječni MAE
za algoritme uključene u kalibraciju. Dobivene parametre treba provjeriti
na odvojenom skupu slika prije promjene zadanih vrijednosti.

Pokretanje: python -m evaluacija.kalibracija [--target localization|boundary|both]"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    # Dodaj korijen projekta za izravno pokretanje skripte iz IDE-a.
    sys.path.insert(0, str(_PROJECT_ROOT))

from algoritmi import common, rgb_euklidska, lab_euklidska, kmeans_euklidska
from evaluacija.oznacavanje import load_ground_truth
from evaluacija.pokreni_evaluaciju import INPUT_DIR, RESULTS_DIR, evaluate_algorithm, write_csv, _fmt

ALGORITHMS = [rgb_euklidska, lab_euklidska, kmeans_euklidska]
EDGE_BOOST_ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
CURVE_FIT_MAD_K_GRID = [2.0, 2.5, 3.0, 4.0]
MIN_ASPECT_RATIO_GRID = [1.0, 1.15, 1.3]
MIN_COVERAGE_RATIO_GRID = [0.12, 0.18, 0.22, 0.28]
MIN_EDGE_DENSITY_GRID = [0.01, 0.02, 0.03]


def combined_mae(ground_truth, input_dir) -> float:
    """Vraća prosječni MAE triju algoritama u postotnim bodovima.

    Koristi trenutačne parametre granice iz modula common."""
    errors = []
    for module in ALGORITHMS:
        summary, *_rest = evaluate_algorithm(module, ground_truth, input_dir)
        if not np.isnan(summary["MAE_%"]):
            errors.append(summary["MAE_%"])
    return float(np.mean(errors)) if errors else float("nan")


def combined_f1(ground_truth, input_dir) -> float:
    """Vraća prosječni F1 triju algoritama uz trenutačne filtre lokalizacije.

    F1 uzima u obzir i preciznost i odziv, pa samo popuštanje pragova
    ne mora poboljšati rezultat."""
    scores = []
    for module in ALGORITHMS:
        summary, *_rest = evaluate_algorithm(module, ground_truth, input_dir)
        p, r = summary["preciznost"], summary["odziv"]
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            continue
        scores.append(2 * p * r / (p + r))
    return float(np.mean(scores)) if scores else float("nan")


def _load_ground_truth_or_exit():
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("Nema ground truth podataka. Prvo pokreni: python -m evaluacija.oznacavanje")
        return None

    n_boxes = sum(len(v) for v in ground_truth.values())
    print(f"Ground truth: {len(ground_truth)} slika, {n_boxes} označenih boca.")
    if n_boxes < 10:
        print(
            f"UPOZORENJE: samo {n_boxes} označenih boca - rezultat je vrlo vjerojatno "
            "preprilagođen (overfit) ovom malom skupu. Uzmi ga kao orijentaciju, ne kao "
            "konačan odabir dok se skup ne proširi.\n"
        )
    return ground_truth


def calibrate_boundary(ground_truth) -> None:
    original_alpha = common.EDGE_BOOST_ALPHA
    original_mad_k = common.CURVE_FIT_MAD_K

    rows: List[dict] = []
    combos = list(itertools.product(EDGE_BOOST_ALPHA_GRID, CURVE_FIT_MAD_K_GRID))
    print(f"\n[granica tekućine] Testiram {len(combos)} kombinacija (MAE cilj)...\n")
    try:
        for alpha, mad_k in combos:
            common.EDGE_BOOST_ALPHA = alpha
            common.CURVE_FIT_MAD_K = mad_k
            mae = combined_mae(ground_truth, INPUT_DIR)
            rows.append({"EDGE_BOOST_ALPHA": alpha, "CURVE_FIT_MAD_K": mad_k, "prosj_MAE_%": mae})
            print(f"  EDGE_BOOST_ALPHA={alpha:.1f}  CURVE_FIT_MAD_K={mad_k:.1f}  ->  MAE={_fmt(mae)}%")
    finally:
        common.EDGE_BOOST_ALPHA = original_alpha
        common.CURVE_FIT_MAD_K = original_mad_k

    rows.sort(key=lambda r: (np.inf if np.isnan(r["prosj_MAE_%"]) else r["prosj_MAE_%"]))
    print("\n=== [granica tekućine] 5 najboljih kombinacija (rastuće po MAE) ===")
    for r in rows[:5]:
        print(
            f"  EDGE_BOOST_ALPHA={r['EDGE_BOOST_ALPHA']:.1f}  "
            f"CURVE_FIT_MAD_K={r['CURVE_FIT_MAD_K']:.1f}  MAE={_fmt(r['prosj_MAE_%'])}%"
        )
    print(
        f"\nTrenutni default: EDGE_BOOST_ALPHA={original_alpha}, "
        f"CURVE_FIT_MAD_K={original_mad_k} (nepromijenjen - ovaj alat samo predlaže)."
    )
    write_csv(RESULTS_DIR / "kalibracija_granice.csv", rows)
    print(f"Spremljeno: {RESULTS_DIR / 'kalibracija_granice.csv'}")


def calibrate_localization(ground_truth) -> None:
    original_aspect = common.MIN_ASPECT_RATIO
    original_coverage = common.MIN_COVERAGE_RATIO
    original_density = common.MIN_EDGE_DENSITY

    rows: List[dict] = []
    combos = list(itertools.product(
        MIN_ASPECT_RATIO_GRID, MIN_COVERAGE_RATIO_GRID, MIN_EDGE_DENSITY_GRID
    ))
    print(f"\n[lokalizacija] Testiram {len(combos)} kombinacija (F1 cilj, ovo traje dulje)...\n")
    try:
        for aspect, coverage, density in combos:
            common.MIN_ASPECT_RATIO = aspect
            common.MIN_COVERAGE_RATIO = coverage
            common.MIN_EDGE_DENSITY = density
            f1 = combined_f1(ground_truth, INPUT_DIR)
            rows.append({
                "MIN_ASPECT_RATIO": aspect, "MIN_COVERAGE_RATIO": coverage,
                "MIN_EDGE_DENSITY": density, "prosj_F1": f1,
            })
            print(
                f"  MIN_ASPECT_RATIO={aspect:.2f}  MIN_COVERAGE_RATIO={coverage:.2f}  "
                f"MIN_EDGE_DENSITY={density:.2f}  ->  F1={_fmt(f1)}"
            )
    finally:
        common.MIN_ASPECT_RATIO = original_aspect
        common.MIN_COVERAGE_RATIO = original_coverage
        common.MIN_EDGE_DENSITY = original_density

    rows.sort(key=lambda r: (-np.inf if np.isnan(r["prosj_F1"]) else -r["prosj_F1"]))
    print("\n=== [lokalizacija] 5 najboljih kombinacija (padajuće po F1) ===")
    for r in rows[:5]:
        print(
            f"  MIN_ASPECT_RATIO={r['MIN_ASPECT_RATIO']:.2f}  "
            f"MIN_COVERAGE_RATIO={r['MIN_COVERAGE_RATIO']:.2f}  "
            f"MIN_EDGE_DENSITY={r['MIN_EDGE_DENSITY']:.2f}  F1={_fmt(r['prosj_F1'])}"
        )
    print(
        f"\nTrenutni default: MIN_ASPECT_RATIO={original_aspect}, "
        f"MIN_COVERAGE_RATIO={original_coverage}, MIN_EDGE_DENSITY={original_density} "
        "(nepromijenjen - ovaj alat samo predlaže)."
    )
    write_csv(RESULTS_DIR / "kalibracija_lokalizacije.csv", rows)
    print(f"Spremljeno: {RESULTS_DIR / 'kalibracija_lokalizacije.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["localization", "boundary", "both"], default="localization",
        help="koji parametri se kalibriraju (default: localization - trenutno uže grlo)",
    )
    args = parser.parse_args()

    ground_truth = _load_ground_truth_or_exit()
    if ground_truth is None:
        return

    if args.target in ("localization", "both"):
        calibrate_localization(ground_truth)
    if args.target in ("boundary", "both"):
        calibrate_boundary(ground_truth)


if __name__ == "__main__":
    main()
