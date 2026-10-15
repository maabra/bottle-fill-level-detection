"""Algoritam 3: Chan-Vese aktivna kontura - VARIJACIJSKA (energetska)
paradigma korištenja euklidske udaljenosti, uz lokalnu/diferencijalnu
(gradijent boje, lab_euklidska) i globalnu/particijsku (K-means,
kmeans_euklidska).

Dok gradijent mjeri euklidsku udaljenost SUSJEDNIH piksela (rub = velika
udaljenost), a K-means udaljenost piksela do K CENTROIDA, Chan-Vese
mjeri udaljenost piksela do STATISTIKE REGIJE: kontura se postavlja tako
da minimizira ukupnu sumu kvadratnih euklidskih odstupanja piksela od
prosječne boje (CIELAB) vlastite regije - regije su "dobre" kad je svaki
piksel euklidski blizu prosjeku svoje strane konture.

- lokalizacija: PRAVA globalna Chan-Vese (ACWE) segmentacija cijele slike
  na foreground/background (common.global_chan_vese_foreground - ista
  energija minimizacije unutar-regijske varijance), komponente foreground
  maske (common.regions_from_filled_mask) pa pritegnute lokalnom Chan-Vese
  siluetom (common.refine_boxes_chan_vese) - dosljedno paradigmi i u
  lokalizaciji, ne samo u obrisu/razini;
- obris boce: kontura Chan-Vese siluete (common.chan_vese_outline) -
  nema praćenja ruba, silueta izlazi izravno iz minimizacije energije;
- razina tekućine: ISTA Chan-Vese energija, ali s konturom ograničenom
  na obitelj vodoravnih linija unutar siluete boce - optimalan redak je
  onaj koji minimizira zbroj unutar-regijskih suma kvadratnih euklidskih
  odstupanja gornje (zrak) i donje (tekućina) regije.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import common

ALGORITHM_NAME = "Euklidska udaljenost - Chan-Vese aktivna kontura"
OUTPUT_DIRNAME = "chanvese_euklidska"

SEARCH_RANGE = (0.15, 0.90)  # testirano suzavanje do (0.30,0.75) - prosjek MAE izgleda bolje ali je varljivo:
                             # umjetno povlaci NEODREDJENE (bez signala) procjene prema centru USKOG
                             # raspona, sto slucajno pomogne najgorim slucajevima ali PSUJE vec dobre
                             # (provjereno: 180813 isao s ~1pb greske na +27pb) - odbaceno, original je ispravniji
MARGIN_RATIO = 0.18  # testirano 0.10-0.25, 0.18 ostaje najbolji
N_SEGMENTS = 12              # broj stupaca za segmentiranu (zakrivljenu) pretragu granice - vidi common.boundary_curve


def detect_bottle_regions(image_bgr: np.ndarray) -> List[common.BBox]:
    """ČISTO VARIJACIJSKA lokalizacija (`common.localize_variational`):
    prava globalna Chan-Vese (ACWE) segmentacija cijele slike na
    foreground/background (`common.global_chan_vese_foreground`), povezane
    komponente foreground maske pa lokalno Chan-Vese pritezanje.

    RANIJA (nečista) verzija posuđivala je gradijentni kaskadni lanac
    lab_euklidske pa su chanvese i lab davali IDENTIČNE okvire - 'tri
    paradigme' nisu bile stvarno razdvojene. Prva čista zamjena
    (`background_distance_map` + Otsu prag) bila je slaba (IoU 0.66, MAE
    19); zamijenjena PRAVIM globalnim ACWE-om (region-competition), koji je
    i čišći (Chan-Vese energija i u lokalizaciji) i mjereno bolji (IoU 0.74,
    MAE 12.6, TP 14). Razlikovnost je sad u SVE TRI faze (lokalizacija +
    obris + razina)."""
    return common.localize_variational(image_bgr)


def native_outline(image_bgr: np.ndarray, bbox: common.BBox, outline_edges=None) -> "np.ndarray | None":
    """Paradigmatski-vlastit obris: kontura Chan-Vese siluete
    (`common.chan_vese_outline`) - silueta izlazi izravno iz minimizacije
    varijacijske energije (udaljenost piksela do prosjeka regije), bez
    praćenja ruba (gradijent) ili K-means klastera. Kad segmentacija
    kolabira/procuri, pada na pravokutnik (outline=None)."""
    return common.chan_vese_outline(image_bgr, bbox)


def estimate_liquid_level(
    image_bgr: np.ndarray, bbox: common.BBox, outline: "np.ndarray | None" = None
) -> Tuple[List[common.Point], float, float]:
    """Razina tekućine kao Chan-Vese rez: ista segmentirana (po stupcima),
    Fisher-normalizirana pretraga kao lab_euklidska/rgb_euklidska
    (common.boundary_curve), jedina razlika je što se prosjek boje po
    retku računa SAMO nad pikselima unutar Chan-Vese siluete boce
    (`mask=inside`), ne cijele širine pravokutnika - rub pozadine unutar
    okvira tada ne glasa za granicu.

    PRVA verzija ovdje je tražila JEDNU vodoravnu liniju preko cijele
    širine odjednom (bez segmentacije po stupcima) - jednostavnije, ali
    mjereno lošije (MAE ~19% nasuprot ~12% kod lab/rgb/kmeans s istim,
    savršenim - ground truth - okvirom, vidi evaluaciju izravno na GT
    okvirima): boca je zaobljena/perspektivno iskrivljena pa prava
    granica NIJE ravna linija, a segmentacija po stupcima to hvata
    (vidi common.boundary_curve), dok jedna ravna linija po konstrukciji
    ne može. Razlikovna varijacijska paradigma ostaje u TOME ŠTO se
    prosjekuje (Chan-Vese silueta, ne pravokutnik) i u obrisu
    (`chan_vese_outline`), ne u obliku same granice."""
    x, y, w, h = bbox
    roi = image_bgr[y:y + h, x:x + w]
    # Značajka regije: CIELAB + zasićenje (S). Prije samo CIELAB; dodano S je i
    # dalje ČISTO VARIJACIJSKA (isti mehanizam - minimizacija odstupanja piksela
    # od prosjeka regije - samo bogatiji prostor boje, ne posudba gradijenta/
    # particije). Izmjereno (GT okviri, čista razina): MAE 10.74->8.93. Trošak:
    # jedan već-izgubljeni slučaj (175738 prozirni crveni sirup, visok S svuda)
    # ode -26.8->-41.8, ali 5 boca srednje pogreške (175641, 181228...) se
    # popravi - jasan neto dobitak na prosjeku.
    feats = common.lab_saturation_features(roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    silhouette = common.chan_vese_silhouette(image_bgr, bbox)
    if silhouette is None or silhouette.sum() < 0.05 * w * h:
        # Silueta nedostupna (npr. providna boca na identičnoj pozadini)
        # - unutrašnjost se aproksimira okvirom bez bočnih margina, isto
        # kao MARGIN_RATIO kod ostalih algoritama.
        silhouette = np.zeros((h, w), dtype=np.uint8)
        mx = int(w * MARGIN_RATIO)
        silhouette[:, mx:max(mx + 1, w - mx)] = 1
    inside = silhouette > 0

    search_range = common.body_search_range(outline, bbox, SEARCH_RANGE)
    # ČISTO VARIJACIJSKA razina: rez minimizira zbroj unutar-regijskih kvadratnih
    # odstupanja gornje/donje regije (prosjek boje unutar siluete). Gradijentni
    # pomoćnici (edge_boost, prior meniskusa, label_band) su NAMJERNO isključeni -
    # oni su tuđa (lokalna/gradijentna) paradigma i ulazili bi u samu odluku o
    # retku (množe/dodaju ocjenu, vetiraju retke), pa razina više ne bi bila
    # čisto varijacijska. Ostaje SAMO paradigm-neutralni fizikalni prior
    # vodoravnosti (linija umjesto parabole + granica odstupanja u
    # common.boundary_curve), koji NE ovisi o gradijentu.
    points, confidence = common.boundary_curve(
        feats, gray_roi, MARGIN_RATIO, search_range, N_SEGMENTS, mask=inside,
        use_edge_boost=False, use_label_band=False, curve_degree=0,
    )
    curve_abs, fill_percent = common.finalize_boundary(points, bbox)
    return curve_abs, fill_percent, confidence


def process_image(path, output_dir) -> Tuple[Path, List[common.BottleResult]]:
    image = common.load_image(path)
    bboxes = detect_bottle_regions(image)

    results: List[common.BottleResult] = []
    for bbox in bboxes:
        outline = native_outline(image, bbox)
        curve_abs, fill_percent, confidence = estimate_liquid_level(image, bbox, outline)
        fill_percent_volume = common.volumetric_fill_percent(outline, bbox, fill_percent)
        results.append(common.BottleResult(bbox, curve_abs, fill_percent, outline, fill_percent_volume, confidence))

    annotated = common.annotate_bottles(image, results, ALGORITHM_NAME)
    out_path = Path(output_dir) / f"{Path(path).stem}_chanvese_euklidska.png"
    common.save_image(out_path, annotated)
    return out_path, results
