"""Detekcija boca i razine tekućine K-means klasteriranjem.

Lokalizacija koristi CIELAB, zasićenje i prostorne koordinate, uz
dodatne prolaze po boji i teksturi. Unutar boce koriste se tri klastera;
klaster najzastupljeniji pri dnu tumači se kao tekućina. Granica se traži
iz njegova udjela po segmentima, bez gradijentnog pojačanja."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import common

ALGORITHM_NAME = "Euklidska udaljenost - K-means (CIELAB + S)"
OUTPUT_DIRNAME = "kmeans_euklidska"
K_BOUNDARY = 8  # Broj klastera za kartu rubova obrisa.
K_LOCALIZATION = 5  # Broj klastera u primarnoj lokalizaciji.
K_LOCAL = 3  # Tri klastera mogu razdvojiti tekućinu, zrak i etiketu ili obojeno staklo.
SPATIAL_WEIGHT = 0.5  # Težina položaja u lokalizaciji; pomaže razdvojiti boce slične boje.
MIN_CLUSTER_BOX_AREA_RATIO = 0.12  # Prag površine za velike kandidate iz primarnog klasteriranja.


# Manji kandidati zadržavaju se ako imaju dovoljno unutarnje teksture.
# Zasićenje se ovdje ne zahtijeva jer može biti slabo kod tamnog stakla.
CLUSTER_STRUCT_MIN_AREA_RATIO = 0.045  # donji prag za strukturirane (bocolike) male regije
CLUSTER_STRUCT_MIN_VAR = 0.12  # min. udio teksturiranih piksela (stvarne boce >=0.124, FP-traka 0.045)
SEARCH_RANGE = (0.15, 0.90)  # Širi raspon čuva visoke i niske razine; uži može pomaknuti slabe procjene prema sredini.
MARGIN_RATIO = 0.18  # Bočna margina kao udio širine okvira.
N_SEGMENTS = 12  # Broj segmenata odabran kalibracijom.


def edge_map(image_bgr: np.ndarray) -> np.ndarray:
    """Vraća kartu prijelaza između K-means klastera."""
    labels, _centers = common.kmeans_cluster(common.lab_saturation_features(image_bgr), K_BOUNDARY)
    edge = np.zeros(labels.shape, dtype=np.uint8)
    edge[:, :-1][labels[:, :-1] != labels[:, 1:]] = 255
    edge[:-1, :][labels[:-1, :] != labels[1:, :]] = 255
    return edge


TEXTURE_FALLBACK_K = 2  # Dva klastera razdvajaju glatku pozadinu od teksturiranog tijela boce.
TEXTURE_FALLBACK_MIN_AREA_RATIO = 0.045  # Minimalna površina teksturnog kandidata.


def texture_cluster_candidate_boxes(image_bgr, existing):
    """Dodaje kandidate iz K-means podjele lokalne teksture.

    Teksturirani klaster može izdvojiti prozirne boce na glatkoj
    pozadini. Premali i već pokriveni kandidati odbacuju se."""
    texture = common._raw_variance_map(image_bgr).astype(np.float64)
    labels, centers = common.kmeans_cluster(texture[:, :, None], TEXTURE_FALLBACK_K)
    high = np.flatnonzero(centers[:, 0] > float(np.median(centers[:, 0])))
    mask = common.exclude_skin((np.isin(labels, high).astype(np.uint8) * 255), image_bgr)
    boxes = common.merge_and_dedup_boxes(common.regions_from_binary_edges(mask))
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    # Zadrži teksturni okvir ako nadilazi postojeći djelomični okvir.
    # Naknadno uklanjanje duplikata daje prednost potpunijem okviru.
    return [b for b in boxes
            if b[2] * b[3] >= TEXTURE_FALLBACK_MIN_AREA_RATIO * img_area
            and all(common._contained_ratio(b, e) < common.CONTAINMENT_DEDUP_RATIO for e in existing)]


def _filter_cluster_boxes(boxes, image_bgr, img_area):
    """Zadržava velike kandidate i manje regije s dovoljno teksture.

    Karta teksture računa se tek kad postoji mali kandidat."""
    big = [b for b in boxes if b[2] * b[3] >= MIN_CLUSTER_BOX_AREA_RATIO * img_area]
    small = [b for b in boxes
             if CLUSTER_STRUCT_MIN_AREA_RATIO * img_area <= b[2] * b[3] < MIN_CLUSTER_BOX_AREA_RATIO * img_area]
    if not small:
        return big
    var_map = common.local_variance_map(image_bgr)
    for b in small:
        bx, by, bw, bh = b
        var_cov = (var_map[by:by + bh, bx:bx + bw] > 0).mean()
        if var_cov >= CLUSTER_STRUCT_MIN_VAR:   # tekstura sama razdvaja - vidi napomenu uz konstantu
            big.append(b)
    return big


def detect_bottle_regions(image_bgr: np.ndarray) -> List[common.BBox]:
    """Lokalizira boce iz maski klastera boje i položaja.

    Fini prolaz dopunjuje se grubljim klasteriranjem i prolazom po
    teksturi. Geometrijsko spajanje i filtri uklanjaju fragmente,
    a tekstura iznad okvira pomaže obuhvatiti čep i grlo."""
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    labels, _centers = common.kmeans_cluster(
        common.lab_saturation_spatial_features(blurred, SPATIAL_WEIGHT), K_LOCALIZATION
    )

    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    boxes: List[common.BBox] = []
    for k in range(K_LOCALIZATION):
        mask = (labels == k).astype(np.uint8) * 255
        # Ukloni regije kože prije kontura kako se ruka ne bi spojila s bocom.
        mask = common.exclude_skin(mask, image_bgr)
        boxes.extend(common.regions_from_binary_edges(mask))
    boxes = common.merge_and_dedup_boxes(boxes)
    # Odbaci male glatke fragmente prije dopunskih prolaza.
    boxes = _filter_cluster_boxes(boxes, image_bgr, img_area)

    # Grublji prolaz može zadržati tijelo koje se finim klasteriranjem razlomilo.
    extra_coarse = common.color_cluster_candidate_boxes(image_bgr, boxes, k=3)
    if extra_coarse:
        boxes = sorted(boxes + extra_coarse, key=lambda b: b[0])

    # Teksturni prolaz dopunjuje boce koje prolazi po boji nisu obuhvatili.
    extra_texture = texture_cluster_candidate_boxes(image_bgr, boxes)
    if extra_texture:
        boxes = sorted(boxes + extra_texture, key=lambda b: b[0])

    # Okviri ostaju izvedeni iz klastera, bez Chan-Vese dorade.
    boxes = common.dedup_boxes(list(boxes), max_boxes=len(boxes) or 1)
    # Tekstura iznad tijela može proširiti okvir na čep i grlo.
    return common.extend_boxes_top(image_bgr, boxes)


def compute_outline_edges(image_bgr: np.ndarray) -> np.ndarray:
    """Vraća granice klastera uz uklonjene regije kože."""
    return common.exclude_skin(edge_map(image_bgr), image_bgr)


def native_outline(image_bgr: np.ndarray, bbox: common.BBox, outline_edges: np.ndarray = None) -> "np.ndarray | None":
    """Prati obris na unaprijed izračunatim granicama klastera.

    Ako nema dovoljno signala, vraća None za prikaz okvira."""
    if outline_edges is None:
        outline_edges = compute_outline_edges(image_bgr)
    return common.bottle_outline(bbox, outline_edges)


def estimate_liquid_level(
    image_bgr: np.ndarray, bbox: common.BBox, outline: "np.ndarray | None" = None
) -> Tuple[List[common.Point], float, float]:
    x, y, w, h = bbox
    roi_bgr = image_bgr[y:y + h, x:x + w]
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    labels, _centers = common.kmeans_cluster(common.lab_saturation_features(roi_bgr), K_LOCAL)

    # Klaster najzastupljeniji u donjih 25% okvira smatra se tekućinom.
    # Šira traka ublažava utjecaj sjene i etikete pri dnu. Kod gotovo
    # prazne boce tom trakom može dominirati zrak.
    margin = int(w * MARGIN_RATIO)
    core_labels = labels[:, margin: w - margin] if w - 2 * margin > 4 else labels
    bottom_rows = max(1, int(h * 0.25))
    bottom_cluster = np.bincount(core_labels[-bottom_rows:].ravel()).argmax()

    feature_map = (labels == bottom_cluster).astype(np.float64)[..., None]

    search_range = common.body_search_range(outline, bbox, SEARCH_RANGE)
    # Granica se traži iz udjela tekućinskog klastera bez gradijentnih dodataka.
    # Uklapa se konstantna razina; maskiranje nepouzdanim obrisom izostavljeno je.
    points, confidence = common.boundary_curve(
        feature_map, gray_roi, MARGIN_RATIO, search_range, N_SEGMENTS,
        use_edge_boost=False, use_label_band=False, curve_degree=0,
    )
    curve_abs, fill_percent = common.finalize_boundary(points, bbox)
    return curve_abs, fill_percent, confidence


def process_image(path, output_dir) -> Tuple[Path, List[common.BottleResult]]:
    image = common.load_image(path)
    bboxes = detect_bottle_regions(image)
    outline_edges = compute_outline_edges(image)

    results: List[common.BottleResult] = []
    for bbox in bboxes:
        outline = native_outline(image, bbox, outline_edges)
        curve_abs, fill_percent, confidence = estimate_liquid_level(image, bbox, outline)
        fill_percent_volume = common.volumetric_fill_percent(outline, bbox, fill_percent)
        results.append(common.BottleResult(bbox, curve_abs, fill_percent, outline, fill_percent_volume, confidence))

    annotated = common.annotate_bottles(image, results, ALGORITHM_NAME)
    out_path = Path(output_dir) / f"{Path(path).stem}_kmeans_euklidska.png"
    common.save_image(out_path, annotated)
    return out_path, results
