"""Algoritam 1: detekcija ruba boce i razine tekućine pomoću euklidske
udaljenosti u RGB prostoru boja (+ zasićenje boje kao dodatna dimenzija).

I lokalizacija boce i granica tekućine grade se na istoj podlozi -
gradijentu dobivenom euklidskom udaljenošću susjednih piksela u RGB
prostoru:
- lokalizacija: magnituda gradijenta se Otsu pragom pretvara u binarnu
  kartu rubova iz koje se izvlače konture oblika boce (vidi
  `common.regions_from_binary_edges` za detalje o spajanju fragmentiranih
  kontura prije filtriranja po obliku);
- razina tekućine: širina okvira dijeli se na više stupaca (boca je 3D
  objekt pa granica nije nužno ravna crta), a u svakom stupcu nezavisno
  se traži redak koji najbolje dijeli [R, G, B, S] vektore na gornju i
  donju skupinu s maksimalnom euklidskom udaljenošću (poopćeni Otsu
  prag), potvrđen stvarnim vertikalnim gradijentom i s isključenim
  recima etikete iz pretrage (vidi `common.boundary_curve`).
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import common

ALGORITHM_NAME = "Euklidska udaljenost - gradijent boje (RGB, ablacija)"
OUTPUT_DIRNAME = "rgb_euklidska"

SEARCH_RANGE = (0.15, 0.90)  # testirano suzavanje do (0.30,0.75) - prosjek MAE izgleda bolje ali je varljivo:
                             # umjetno povlaci NEODREDJENE (bez signala) procjene prema centru USKOG
                             # raspona, sto slucajno pomogne najgorim slucajevima ali PSUJE vec dobre
                             # (provjereno: 180813 isao s ~1pb greske na +27pb) - odbaceno, original je ispravniji
MARGIN_RATIO = 0.18  # testirano 0.10-0.25, 0.18 ostaje najbolji
N_SEGMENTS = 12  # kalibrirano grid-searchom (9-21); prosjek MAE sva 4 algoritma pao s 10.19% (9) na 9.77%


def edge_map(image_bgr: np.ndarray) -> np.ndarray:
    """Karta rubova: magnituda gradijenta dobivena euklidskom udaljenošću
    boje susjednih piksela u RGB prostoru (svaka boja = jedna dimenzija)."""
    img = image_bgr.astype(np.float64)

    gx = np.zeros(img.shape[:2])
    gy = np.zeros(img.shape[:2])

    diff_x = img[:, 2:, :] - img[:, :-2, :]
    diff_y = img[2:, :, :] - img[:-2, :, :]

    gx[:, 1:-1] = np.linalg.norm(diff_x, axis=2)
    gy[1:-1, :] = np.linalg.norm(diff_y, axis=2)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val * 255
    return magnitude.astype(np.uint8)


def _boxes_and_binary(image_bgr: np.ndarray):
    """Vraća (okviri, binarna karta koja ih je dala) - ČISTO GRADIJENTNA
    lokalizacija (`common.localize_gradient`), ovdje s RGB gradijentom
    (`edge_map`). Isti pipeline kao `lab_euklidska` (ovo je ablacija u RGB
    prostoru boje) - jedina razlika je prostor u kojem se računa gradijent.
    Binarna karta se vraća za obris (`common.bottle_outline`)."""
    return common.localize_gradient(image_bgr, edge_map)


def detect_bottle_regions(image_bgr: np.ndarray) -> List[common.BBox]:
    boxes, _binary = _boxes_and_binary(image_bgr)
    return boxes


def compute_outline_edges(image_bgr: np.ndarray, binary: np.ndarray = None) -> np.ndarray:
    """Karta rubova za `common.bottle_outline` - dodatno fuzionira sirovi
    (nezamućeni) signal zasićenja radi finijeg detalja preko `binary`
    (koja već ima isključenu boju kože), pa se isključivanje kože ponovno
    primjenjuje da se ruka ne vrati kroz ovaj dodatni izvor (vidi
    common.saturation_edge_map/exclude_skin). Javna je (ne `_`-prefiks) da
    je i evaluacijska skripta može pozvati bez ponovnog otkrivanja logike
    iz `process_image` - `binary` je opcionalan jer ga `process_image` već
    ima izračunatog (iz `_boxes_and_binary`), a evaluacija ne."""
    if binary is None:
        _, binary = _boxes_and_binary(image_bgr)
    return common.exclude_skin(
        cv2.bitwise_or(binary, common.saturation_edge_map(image_bgr)), image_bgr
    )


def native_outline(image_bgr: np.ndarray, bbox: common.BBox, outline_edges) -> "np.ndarray | None":
    """Paradigmatski-vlastit obris: gradijentno praćenje ruba
    (`common.bottle_outline`), isto kao lab_euklidska (ovo je ablacija u
    RGB prostoru boje). Bez Chan-Vese/klaster fallbacka - kad praćenje ne
    uspije, pada na pravokutnik."""
    return common.bottle_outline(bbox, outline_edges)


def _feature_map(roi_bgr: np.ndarray) -> np.ndarray:
    saturation = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return np.concatenate(
        [roi_bgr.astype(np.float64), saturation[:, :, None].astype(np.float64)],
        axis=2,
    )


def estimate_liquid_level(
    image_bgr: np.ndarray, bbox: common.BBox, outline: "np.ndarray | None" = None
) -> Tuple[List[common.Point], float, float]:
    x, y, w, h = bbox
    roi = image_bgr[y:y + h, x:x + w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    feature_map = _feature_map(roi)

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
    out_path = Path(output_dir) / f"{Path(path).stem}_rgb_euklidska.png"
    common.save_image(out_path, annotated)
    return out_path, results
