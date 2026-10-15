"""Interaktivni alat za ručno označavanje stvarne (ground truth) razine
tekućine u bocama - bez ovoga nema objektivne osnove za usporedbu tri
algoritma (vidi `pokreni_evaluaciju.py`).

Tijek rada za svaku sliku iz ulaznog foldera:
1. Prozor za odabir okvira (`cv2.selectROIs`): povlačenjem miša označi
   okvir oko svake boce na slici, ENTER/SPACE potvrđuje okvir i nastavlja
   sljedeći, ESC završava odabir za tu sliku (ili je preskače ako nijedan
   okvir nije povučen).
2. Za svaki povučeni okvir otvara se uvećani isječak te boce: pomicanjem
   miša crta se crvena vodoravna crta uz uživo prikazan postotak, lijevi
   klik je postavlja, ENTER potvrđuje i sprema oznaku, ESC preskače tu
   bocu (bez spremanja).

Rezultat se nakon svake slike odmah upisuje u `evaluacija/ground_truth.json`
(da prekid alata ne izgubi već označene slike):

    {
      "IMG_6205.webp": [
        {"bbox": [x, y, w, h], "level_y": 812}
      ]
    }

`bbox` i `level_y` su u koordinatama originalne (neskalirane) slike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    # Omogućuje pokretanje i izravno (npr. IDE "Run" gumb), ne samo kao
    # `python -m evaluacija.oznacavanje` iz korijena projekta - bez ovoga
    # `algoritmi` paket (susjed ovog foldera) nije vidljiv jer Python u tom
    # slučaju na sys.path stavi samo folder same skripte (`evaluacija/`).
    sys.path.insert(0, str(_PROJECT_ROOT))

from algoritmi import common

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "boce"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.json"

MAX_DISPLAY_DIM = 1000       # najveća dimenzija prozora za odabir okvira
LEVEL_DISPLAY_HEIGHT = 750   # ciljna visina prozora za odabir granice

ROI_WINDOW = "1) Oznaci okvir(e) oko boce - ENTER/SPACE po okviru, ESC kad gotovo"
LEVEL_WINDOW = "2) Klikni na granicu tekucine - ENTER potvrdi, ESC preskoci bocu"


def load_ground_truth() -> Dict[str, List[dict]]:
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ground_truth(data: Dict[str, List[dict]]) -> None:
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _display_scale(image: np.ndarray, max_dim: float) -> float:
    h, w = image.shape[:2]
    return min(1.0, max_dim / max(h, w))


def _select_level_y(image_bgr: np.ndarray, bbox: common.BBox) -> Optional[int]:
    """Prikaže uvećani isječak boce i vrati apsolutnu y-koordinatu (u
    originalnoj slici) klika koji označava granicu tekućine, ili None ako
    je korisnik pritisnuo ESC (boca se preskače)."""
    x, y, w, h = bbox
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, image_bgr.shape[1]), min(y + h, image_bgr.shape[0])
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    scale = LEVEL_DISPLAY_HEIGHT / crop.shape[0] if crop.shape[0] else 1.0
    scale = min(scale, 4.0)  # ne povećavaj pretjerano male okvire
    disp = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    state = {"y": disp.shape[0] // 2, "clicked": False}

    def on_mouse(event, mx, my, _flags, _userdata):
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            state["y"] = int(np.clip(my, 0, disp.shape[0] - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicked"] = True

    cv2.namedWindow(LEVEL_WINDOW)
    cv2.setMouseCallback(LEVEL_WINDOW, on_mouse)

    result_y = None
    try:
        while True:
            frame = disp.copy()
            line_y = state["y"]
            fill_pct = 100.0 * (frame.shape[0] - line_y) / frame.shape[0]
            color = (0, 0, 255) if state["clicked"] else (0, 165, 255)
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), color, 2)
            cv2.putText(
                frame, f"{fill_pct:.0f}%", (8, max(20, line_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )
            hint = "ENTER = potvrdi" if state["clicked"] else "klikni na granicu tekucine"
            cv2.putText(
                frame, f"{hint}   ESC = preskoci bocu",
                (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow(LEVEL_WINDOW, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and state["clicked"]:  # ENTER
                result_y = state["y"]
                break
            if key == 27:  # ESC
                result_y = None
                break
    finally:
        cv2.destroyWindow(LEVEL_WINDOW)

    if result_y is None:
        return None
    return y0 + int(round(result_y / scale))


def annotate_image(image: np.ndarray) -> List[dict]:
    scale = _display_scale(image, MAX_DISPLAY_DIM)
    disp = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else image.copy()
    )

    cv2.namedWindow(ROI_WINDOW, cv2.WINDOW_NORMAL)
    boxes = cv2.selectROIs(ROI_WINDOW, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(ROI_WINDOW)

    annotations: List[dict] = []
    if boxes is None or len(boxes) == 0:
        return annotations

    for bx, by, bw, bh in boxes:
        if bw <= 0 or bh <= 0:
            continue
        orig_box = (
            int(round(bx / scale)), int(round(by / scale)),
            int(round(bw / scale)), int(round(bh / scale)),
        )
        level_y = _select_level_y(image, orig_box)
        if level_y is None:
            continue
        annotations.append({"bbox": list(orig_box), "level_y": level_y})

    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--redo", action="store_true",
        help="ponovno označi i slike koje već imaju oznake u ground_truth.json",
    )
    args = parser.parse_args()

    image_paths = common.list_images(args.input_dir)
    if not image_paths:
        print(f"Nema podržanih slika u folderu: {args.input_dir}")
        return

    ground_truth = load_ground_truth()
    print(f"Pronađeno {len(image_paths)} slika u '{args.input_dir.name}'.")
    print(f"Postojeće oznake: {len(ground_truth)} slika (u {GROUND_TRUTH_PATH.name}).")
    print(
        "\n1) Povuci okvir oko svake boce (ENTER/SPACE po okviru, ESC kad gotovo)."
        "\n2) Klikni na granicu tekućine u svakom okviru (ENTER potvrdi, ESC preskače bocu).\n"
    )

    try:
        for path in image_paths:
            if path.name in ground_truth and not args.redo:
                print(f"{path.name}: već označeno, preskačem (koristi --redo za ponovno)")
                continue

            image = common.load_image(path)
            annotations = annotate_image(image)

            if annotations:
                ground_truth[path.name] = annotations
                save_ground_truth(ground_truth)
                print(f"{path.name}: spremljeno {len(annotations)} oznaka/e")
            else:
                print(f"{path.name}: nema spremljenih oznaka (preskočeno)")
    finally:
        cv2.destroyAllWindows()

    print(f"\nGotovo. Ground truth spremljen u: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
