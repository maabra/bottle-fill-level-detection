"""Zajedničke funkcije koje dijele sva tri algoritma: učitavanje slika
različitih formata, pronalaženje boca na slici, praćenje granice tekućine
te iscrtavanje rezultata."""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]


class BottleResult(NamedTuple):
    """Rezultat obrade jedne boce. `outline` je stvaran obris boce
    izmjeren iz ruba na slici (vidi `bottle_outline`) ili None ako nije
    pronađeno dovoljno pouzdanog ruba (npr. posve providna boca na sličnoj
    pozadini) - pozivni kod (crtanje) tada pada natrag na pravokutnik
    `bbox`. `fill_percent_volume` je postotak VOLUMENA (vidi
    `volumetric_fill_percent`) - kod bačvastih/suženih boca (grlo) razlikuje
    se od linearnog `fill_percent`; jednak mu je kad obris nije dostupan.
    `confidence` je pouzdanost granice tekućine (vidi `best_split_index`) -
    ispod `CONFIDENCE_THRESHOLD` znači da u dopuštenom rasponu pretrage
    nema stvarne granice boje (providna boca bez signala, ili je stvarna
    razina izvan raspona), pa `fill_percent` treba tumačiti kao nagađanje,
    ne mjerenje."""

    bbox: BBox
    curve: List[Point]
    fill_percent: float
    outline: Optional[np.ndarray] = None
    fill_percent_volume: Optional[float] = None
    confidence: float = 1.0


# Kalibrirano na stvarnim podacima (evaluacija/rezultati/detaljno.csv, stupac
# "pouzdanost"): unutar sparenih (TP) boca pouzdanost NE korelira pouzdano s
# veličinom pogreške (raspon ~1.8-30 kod stvarnih, čak i pogrešnih, rezova -
# omjer mjeri POSTOJI li ikakva boja-tranzicija, ne je li to PRAVA tranzicija
# tekućine, npr. rub grafike na etiketi i dalje da visok omjer). Zato prag
# NIJE postavljen da razdvoji "točno" od "netočno" (to ne uspijeva ni kod
# jednog testiranog praga), nego da uhvati potpuno DEGENERIRAN slučaj - kad
# u cijelom rasponu pretrage nema NIKAKVE boja-razlike (npr. providna boca
# na identičnoj pozadini, ili okvir koji je izgubio vrh boce): takvi slučajevi
# mjereno imaju pouzdanost ~0.00-0.10, dok svaki slučaj sa ikakvim stvarnim
# signalom (točnim ili pogrešnim) ima >=1.78 - velik, jasan razmak. 1.0 sjedi
# udobno u tom praznom razmaku.
CONFIDENCE_THRESHOLD = 1.0


# --------------------------------------------------------------------------
# Učitavanje / spremanje slika
# --------------------------------------------------------------------------

def list_images(folder) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Ne mogu učitati sliku: {path}")
    return image


def ensure_dir(folder) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)


def save_image(path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image_bgr)


# --------------------------------------------------------------------------
# Lokalizacija boca
# --------------------------------------------------------------------------
# Parametri geometrijskog filtriranja kandidata (nazvani umjesto "magičnih
# brojeva" radi čitljivosti i lakšeg podešavanja).
MIN_FRAGMENT_AREA_RATIO = 0.0004   # donji prag prije spajanja - uklanja samo šum
MIN_BOTTLE_AREA_RATIO = 0.02      # donji prag nakon spajanja - stvarna boca. Bilo 0.008 (kalibrirano
    # na starom skupu od 17 VELIKIH boca), ali na punom setu (65 boca, složene kuhinjske scene) je
    # 0.008 propuštao mnoštvo sitnih fragmenata pozadine (fuge pločica, rubovi, sitni predmeti) kao
    # lažne pozitive. Najmanja STVARNA boca u punom setu ima površinu 0.037, pa 0.02 (sigurno ispod)
    # reže sitni šum a ne dira stvarne boce. Izmjereno (loc eval, 65 boca): FP lab 33->14, rgb 32->18,
    # chanvese 29->12; preciznost lab 0.49->0.70, chanvese 0.48->0.68; odziv ~nepromijenjen; pomaže i
    # jednostavnim scenama (lab jed FP 9->1). 0.03 još čistiji ali gubi po 1 boca (lab/kmeans) - 0.02 knee.
MAX_BOTTLE_AREA_RATIO = 0.85
MIN_ASPECT_RATIO = 1.15            # boca je uspravna i uža nego što je visoka
MAX_ASPECT_RATIO = 6.0             # ali ne izrazito tanka "iver" nastala spajanjem šuma
MAX_WIDTH_RATIO = 0.5
MIN_EDGE_DENSITY = 0.03            # mora postojati stvarni obris/tekstura, ne prazna pozadina
MIN_COVERAGE_RATIO = 0.18          # okvir mora uglavnom biti "boca", ne slučajan spoj fragmenata
# MIN_COVERAGE_RATIO/MIN_EDGE_DENSITY kalibrirani grid-searchom (F1 cilj)
# nad stvarnim ground truth (34 slike/61 boca) - vidi
# evaluacija/kalibracija.py --target localization i
# evaluacija/rezultati/kalibracija_lokalizacije.csv.
# Snizivanje MAX_BOTTLE_AREA_RATIO (0.85->0.45 i 0.85->0.65, prema GT
# statistici gdje najveća stvarna boca zauzima 0.31 udjela slike) je
# ISPROBANO i vraćeno: labav detektirani okvir preko GT boce smije s
# IoU>=0.4 biti višestruko veći od nje, pa su obje strože granice rezale
# i ispravne detekcije (TP 36->33 na RGB) uz zanemariv pad lažnih
# pozitiva. Jednako je ISPROBANO i vraćeno pooštravanje
# MIN_BOTTLE_AREA_RATIO (0.008->0.003) i MIN_ASPECT_RATIO (1.15->1.6):
# sitni fragmenti su eksplodirali kao lažni pozitivi (preciznost
# 0.50->0.33), a detektirani okvir zna biti zdepastiji od stvarne boce
# pa je stroži omjer rezao i ispravne detekcije.
BORDER_MARGIN_PX = 2
# Boja kože (YCrCb raspon) - ruka koja drži bocu za grlo/čep često stvara
# vlastiti dovoljno jak rub/klaster, prostorno prislonjen uz bocu, pa ga
# `merge_stacked_boxes` spoji u isti okvir. Isključivanje regija boje kože
# PRIJE traženja kontura sprječava to kod izvora (vidi `exclude_skin`).
SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
SKIN_MIN_SATURATION = 60   # HSV zasićenje - sam YCrCb raspon je preširok i pogađa
                           # blijedu/toplo-bijelu pozadinu (studijske fotografije);
                           # prava koža ima primjetnu boju, gotovo bijela pozadina nema
SKIN_MASK_DILATE_PX = 7
# Stvaran obris boce - vidi `bottle_outline`. Prijašnji pokušaji: praćenje
# SVIH kontura cijele mase (nazubljeno - pratilo i unutarnju teksturu),
# fiksni predložak (uvijek uspije, ne prati stvaran oblik), lokalna
# K-means segmentacija (euklidska udaljenost do centroida u CIELAB+S -
# ista metodologija kao `kmeans_euklidska.py`, ali gora od ovoga: providna
# boca nema svoju boju pa se doslovno grupira SA pozadinom iza sebe, što
# u masku unese komade pozadine). Gradijent/rub (lokalni kontrast) pokaže
# se i kod providnog stakla/plastike zbog loma svjetlosti, čak i kad
# boja ne pokazuje ništa - zato praćenje ruba s kontinuitetom
# (`_track_side`) ovdje pobjeđuje unatoč tome što je "manje euklidsko".
BOTTLE_EDGE_MARGIN_PX = 3          # marža od ruba okvira pri traženju ruba prema unutra
BOTTLE_MIN_ROW_COVERAGE = 0.35     # min. udio redaka s pouzdanim rubom da se obrisu vjeruje
BOTTLE_SEED_BAND_RATIO = 0.10      # traka oko sredine visine za robusnu početnu (seed) poziciju
BOTTLE_TRACK_MAX_JUMP_RATIO = 0.05 # najveći dopušten pomak ruba između susjednih redaka (udio širine)
BOTTLE_SMOOTH_PASSES = 3           # broj uzastopnih prolaza pokretnog medijana
BOTTLE_SMOOTH_WINDOW_RATIO = 0.08  # prozor pokretnog medijana kao udio visine okvira - providna
                                    # boca ima kaotičan rub red-po-red (refleksije, nabori), pa
                                    # prozor mora biti širok (postotak visine, ne fiksni piksela)
                                    # da uopće nešto zagladi, ne samo blago ukloni pojedinačni šum
BOTTLE_SMOOTH_WINDOW_MIN = 9       # najmanji dopušteni prozor (za niske/male okvire)
BOTTLE_SIMPLIFY_RATIO = 0.004      # udio opsega - Douglas-Peucker pojednostavljenje poligona


def iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection-over-union dva (x, y, w, h) okvira. Javna je (ne
    `_iou`) jer je koristi i `evaluacija/metrike.py` za sparivanje
    detektiranih okvira sa stvarnima (ground truth)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def merge_stacked_boxes(boxes: Sequence[BBox], passes: int = 4) -> List[BBox]:
    """Spaja okvire koji vjerojatno pripadaju istoj boci, a razdvojeni su u
    više kontura zbog čepa/vrata ili unutarnje teksture (natpis na etiketi,
    nabori plastike, mjehurići) koja fragmentira jedinstvenu konturu boce na
    više manjih. Namjerno se poziva PRIJE filtriranja po obliku/površini
    (vidi `regions_from_binary_edges`) - pojedini fragment sam za sebe
    često ne izgleda kao boca (premalen je ili krivog omjera stranica), pa
    bi ga filtriranje unaprijed odbacilo prije nego što dobije priliku da
    se spoji u cjelovitu konturu."""
    boxes = list(boxes)
    for _ in range(passes):
        boxes.sort(key=lambda b: b[1])
        merged = []
        used = [False] * len(boxes)
        for i, box in enumerate(boxes):
            if used[i]:
                continue
            x, y, w, h = box
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                mx, my, mw, mh = boxes[j]
                x_overlap = max(0, min(x + w, mx + mw) - max(x, mx))
                min_w = min(w, mw)
                if min_w == 0:
                    continue
                x_overlap_ratio = x_overlap / min_w
                vertical_gap = max(0, max(y, my) - min(y + h, my + mh))
                combined_h = max(y + h, my + mh) - min(y, my)
                if x_overlap_ratio > 0.45 and vertical_gap < 0.35 * combined_h:
                    nx, ny = min(x, mx), min(y, my)
                    nw = max(x + w, mx + mw) - nx
                    nh = max(y + h, my + mh) - ny
                    x, y, w, h = nx, ny, nw, nh
                    used[j] = True
            merged.append((x, y, w, h))
        boxes = merged
    return boxes


CONTAINMENT_DEDUP_RATIO = 0.85  # udio manjeg okvira koji mora ležati unutar većeg
                                 # da ga se tretira kao duplikat - vidi dedup_boxes


def _contained_ratio(inner: BBox, outer: BBox) -> float:
    """Udio površine `inner` koji leži unutar `outer` - za razliku od IoU-a
    ne ovisi o veličini `outer`, pa hvata i duplikate vrlo različitih
    veličina (vidi napomenu u `dedup_boxes`)."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    ix1, iy1 = max(ix, ox), max(iy, oy)
    ix2, iy2 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    cw, ch = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inner_area = iw * ih
    return (cw * ch) / inner_area if inner_area else 0.0


def dedup_boxes(boxes: Sequence[BBox], iou_threshold: float = 0.25, max_boxes: int = 6) -> List[BBox]:
    """Uklanja preklapajuće okvire (npr. kad više klastera/segmenata
    predloži gotovo isti okvir), zadržavajući najveći od svake preklapajuće
    skupine. Pooštravanje praga na 0.15 (ideja: ukloniti duplikate oko
    iste boce koji se u evaluaciji broje kao lažni pozitivi) je ISPROBANO
    i vraćeno - okviri stvarno susjednih boca u zagušenim scenama
    preklapaju se više od 0.15, pa je stroži prag rezao i ispravne
    detekcije (TP 36->34 na RGB, 32->30 na K-means varijanti).

    Uz IoU, dodatno se uklanja i okvir GOTOVO POTPUNO (>=CONTAINMENT_DEDUP_RATIO)
    sadržan unutar već zadržanog većeg okvira, čak i kad je IoU nizak - IoU
    sam po sebi to promaši kad se veličine dvaju okvira jako razlikuju (mali
    okvir posve unutar velikog ima nizak IoU jer uniju dominira veliki okvir),
    a upravo je to potpisalo dio stvarnih lažnih pozitiva u evaluaciji
    (kmeans_euklidska, dijagnosticirano na 20260723_181058: labav okvir iz
    klastera i njemu ugniježđen tjesniji, oba prežive IoU dedup jer je
    IoU=0.21 < 0.25)."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[BBox] = []
    for box in boxes:
        duplicate = any(
            iou(box, other) > iou_threshold
            or _contained_ratio(box, other) >= CONTAINMENT_DEDUP_RATIO
            for other in kept
        )
        if not duplicate:
            kept.append(box)
        if len(kept) >= max_boxes:
            break
    kept.sort(key=lambda b: b[0])
    return kept


def merge_and_dedup_boxes(boxes: Sequence[BBox], max_boxes: int = 6) -> List[BBox]:
    """Za slučajeve kad kandidati stižu iz više neovisnih izvora koji mogu
    prekrivati RAZLIČITE dijelove iste boce (npr. K-means lokalizacija, gdje
    tijelo i grlo boce mogu pripasti različitim klasterima pa svaki
    predloži samo dio boce) - prvo geometrijski spaja bliske/preklapajuće
    okvire, a tek onda uklanja preostala preklapanja."""
    return dedup_boxes(merge_stacked_boxes(list(boxes)), max_boxes=max_boxes)


def _close_binary_edges(binary_edges: np.ndarray) -> np.ndarray:
    """Morfološko zatvaranje binarne karte rubova/maske - spaja sitne
    prekide (npr. natpis na etiketi koji prekida obris boce) prije
    izvlačenja kontura u `regions_from_binary_edges` (lokalizacija)."""
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 25))
    closed = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    return cv2.dilate(closed, np.ones((5, 5), np.uint8), iterations=1)


def saturation_edge_map(image_bgr: np.ndarray) -> np.ndarray:
    """Otsu-pragom binarizirana euklidska magnituda gradijenta SAMO S
    (zasićenje) kanala - dodatni, neovisan signal ruba koji se u
    `bottle_outline` fuzionira (logičko ILI) s glavnim rubom svakog
    algoritma (gradijent intenziteta/boje kod RGB/Lab, granice klastera
    kod K-means): granica boce zna biti vidljivija u promjeni zasićenja
    (npr. prijelaz tekućina/zrak, providno staklo/pozadina) nego u
    intenzitetu, ili obrnuto - fuzija dvaju izvora daje pratitelju ruba
    (`_track_side`) više kandidata u svakom retku, pa je manja šansa da
    stvaran rub potpuno nedostaje."""
    saturation = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float64)

    gx = np.zeros(saturation.shape)
    gy = np.zeros(saturation.shape)
    gx[:, 1:-1] = saturation[:, 2:] - saturation[:, :-2]
    gy[1:-1, :] = saturation[2:, :] - saturation[:-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val * 255
    magnitude = magnitude.astype(np.uint8)

    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def exclude_skin(binary_mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Uklanja regije boje kože (YCrCb raspon) iz binarne karte (rubova ili
    klaster maske, iste dimenzije kao `image_bgr`) prije traženja kontura
    u lokalizaciji - ruka koja drži bocu za grlo/čep inače stvara vlastiti
    dovoljno jak rub/klaster koji se, jer je prostorno prislonjen uz bocu,
    spoji u isti okvir (vidi `merge_stacked_boxes`).

    Sam YCrCb raspon je namjerno širok (pokriva razne tonove kože), pa bez
    dodatnog uvjeta pogrešno hvata i blijedu/toplo-bijelu pozadinu (npr.
    studijske fotografije s toplim bijelim balansom) - dijagnosticirano na
    stvarnom promašaju gdje je "koža" pokrivala 98% cijele slike. Zato se
    dodatno traži i stvarna zasićenost boje (HSV S kanal) - koža ima
    primjetnu boju, gotovo bijela/siva pozadina nema, bez obzira što joj
    YCrCb ton slučajno upada u raspon.

    Maska kože se prije primjene malo proširi (dilate) da pokrije i tanku
    prijelaznu zonu oko ruba prstiju, ne samo najizraženiju sredinu.

    Sama boja NIJE dovoljna za razlikovanje: crvenkasto-ružičasta boca
    (npr. prozirna boca crvene tekućine, Cr~157 Cb~117 S~128) upada točno
    u YCrCb+S raspon kože i cijela bi se izbrisala iz svih signala -
    dijagnosticirano na stvarnom promašaju gdje NIJEDAN algoritam nije
    našao bocu jer je 66% njezinog okvira proglašeno kožom. Razdvaja ih
    TOPOLOGIJA, ne boja: ruka koja drži bocu uvijek ulazi u kadar izvana
    (podlaktica dira rub slike), dok bocoliki objekt boje kože stoji
    slobodno u sredini kadra - zato se brišu samo komponente kože
    spojene s rubom slike."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, SKIN_YCRCB_LOWER, SKIN_YCRCB_UPPER)
    saturation = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    skin[saturation < SKIN_MIN_SATURATION] = 0

    num_labels, cc_labels = cv2.connectedComponents(skin, connectivity=8)
    if num_labels > 1:
        border_labels = np.unique(
            np.concatenate([cc_labels[0], cc_labels[-1], cc_labels[:, 0], cc_labels[:, -1]])
        )
        border_labels = border_labels[border_labels != 0]
        skin = np.where(np.isin(cc_labels, border_labels), skin, 0).astype(np.uint8)

    skin = cv2.dilate(skin, np.ones((SKIN_MASK_DILATE_PX, SKIN_MASK_DILATE_PX), np.uint8), iterations=1)
    result = binary_mask.copy()
    result[skin > 0] = 0
    return result


LOCAL_VARIANCE_WINDOW = 9  # veličina prozora za lokalnu varijancu (mora biti neparan)


def _raw_variance_map(image_bgr: np.ndarray, window: int = LOCAL_VARIANCE_WINDOW) -> np.ndarray:
    """Lokalna varijanca (tekstura) sivog kanala, SKALIRANA (0-255) ali BEZ
    Otsu praga - dijeljena osnova za `local_variance_map` (globalni Otsu)
    i `extend_boxes_top` (lokalni Otsu, vidi tamošnju napomenu)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window, window))
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(window, window))
    variance = np.clip(mean_sq - mean ** 2, 0, None)
    std = np.sqrt(variance)
    max_val = std.max()
    if max_val > 0:
        std = std / max_val * 255
    return std.astype(np.uint8)


def local_variance_map(image_bgr: np.ndarray, window: int = LOCAL_VARIANCE_WINDOW) -> np.ndarray:
    """Otsu-pragom binarizirana lokalna varijanca (tekstura) sivog kanala -
    posljednji fallback signal za lokalizaciju (vidi `detect_bottle_regions`
    u svakom algoritamskom modulu) za slike gdje boca gotovo nema
    kontrasta prema pozadini (jednolična bijela/siva studijska pozadina +
    prozirna/svijetla boca), pa NI gradijent NI zasićenje (`saturation_edge_map`)
    ne nađu ništa - dijagnosticirano na stvarnim promašajima u
    evaluaciji (vidi evaluacija/rezultati). Čak i tada boca ima suptilnu
    unutarnju teksturu (odsjaje, rub tekućine, nabore plastike) koje
    jednolična pozadina nema - lokalna varijanca razlikuje "nešto" od
    "ništa" tamo gdje apsolutni kontrast/boja ne uspijeva."""
    std = _raw_variance_map(image_bgr, window)
    _, binary = cv2.threshold(std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


CHAN_VESE_ITERATIONS = 50      # broj iteracija evolucije konture - testirano 50 vs 100, 100 lošije (prekonvergira, curi)
CHAN_VESE_SMOOTHING = 1        # morfoloških open/close prolaza po iteraciji (zakrivljenost/glatkoća)
CHAN_VESE_MARGIN_RATIO = 0.12  # proširenje isječka oko okvira - stvaran rub smije biti malo izvan okvira
CHAN_VESE_INIT_INSET = 0.08    # početna maska: okvir uvučen za ovaj udio (sigurno "unutar" boce) - kalibrirano
                               # (0.05-0.30); manji inset = veće početno sjeme = stabilnija konvergencija
                               # (Chan-Vese MAE 12.0->10.3), 0.08 kompromis prema riziku "curenja" na 0.05
CHAN_VESE_MAX_DIM = 360        # radna rezolucija isječka - dovoljna za siluetu, drži iteracije brzima
CHAN_VESE_MIN_AREA_RATIO = 0.30  # maska manja od ovog udjela okvira = kolaps (npr. providna boca), odbaci
CHAN_VESE_MAX_AREA_RATIO = 0.98  # maska preko cijelog isječka = curenje u pozadinu, odbaci


def _chan_vese_mask(image_bgr: np.ndarray, bbox: BBox):
    """Morfološka Chan-Vese (ACWE) segmentacija isječka oko okvira boce -
    silueta se pronalazi minimizacijom sume KVADRATNIH EUKLIDSKIH
    udaljenosti piksela od prosječne boje (CIELAB) unutar konture i od
    prosječne boje izvan nje: kontura se u svakoj iteraciji na svom rubu
    pomiče tako da svaki piksel završi u regiji čijem je prosjeku
    euklidski bliži, uz morfološko zaglađivanje kao ograničenje
    zakrivljenosti. Za razliku od praćenja ruba redak-po-redak
    (`bottle_outline`), radi nad cijelom regijom odjednom pa ne ovisi o
    kontinuitetu rubnog signala - providna boca s isprekidanim rubom i
    dalje ima konzistentno drugačiju PROSJEČNU boju od pozadine.

    Vraća (maska u koordinatama isječka, (x0, y0) ishodište isječka u
    slici) ili (None, None) ako maska kolabira/procuri (providna boca na
    identičnoj pozadini) - pozivni kod tada pada na postojeće ponašanje."""
    x, y, w, h = bbox
    H, W = image_bgr.shape[:2]
    mx, my = int(w * CHAN_VESE_MARGIN_RATIO), int(h * CHAN_VESE_MARGIN_RATIO)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    crop = image_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 20 or cw < 20:
        return None, None

    scale = 1.0
    if max(ch, cw) > CHAN_VESE_MAX_DIM:
        scale = CHAN_VESE_MAX_DIM / max(ch, cw)
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = crop
    feats = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float64)
    sh, sw = feats.shape[:2]

    bx = int(round((x - x0) * scale))
    by = int(round((y - y0) * scale))
    bw = max(1, int(round(w * scale)))
    bh = max(1, int(round(h * scale)))
    ix = bx + int(bw * CHAN_VESE_INIT_INSET)
    iy = by + int(bh * CHAN_VESE_INIT_INSET)
    iw = max(3, bw - 2 * int(bw * CHAN_VESE_INIT_INSET))
    ih = max(3, bh - 2 * int(bh * CHAN_VESE_INIT_INSET))
    u = np.zeros((sh, sw), dtype=np.uint8)
    u[max(0, iy):min(sh, iy + ih), max(0, ix):min(sw, ix + iw)] = 1
    if u.sum() < 10:
        return None, None

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(CHAN_VESE_ITERATIONS):
        inside = u > 0
        n_in = int(inside.sum())
        if n_in < 10 or n_in > inside.size - 10:
            break
        c1 = feats[inside].mean(axis=0)
        c2 = feats[~inside].mean(axis=0)
        d1 = ((feats - c1) ** 2).sum(axis=2)
        d2 = ((feats - c2) ** 2).sum(axis=2)

        dil = cv2.dilate(u, kernel)
        ero = cv2.erode(u, kernel)
        boundary = dil != ero
        u[boundary & (d1 < d2)] = 1
        u[boundary & (d1 >= d2)] = 0

        for _ in range(CHAN_VESE_SMOOTHING):
            u = cv2.morphologyEx(u, cv2.MORPH_OPEN, kernel)
            u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, kernel)

    # Najveća povezana komponenta - odsijeca eventualne otoke šuma.
    n_labels, labels = cv2.connectedComponents(u)
    if n_labels < 2:
        return None, None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    u = (labels == int(sizes.argmax())).astype(np.uint8)

    box_area_scaled = bw * bh
    mask_area = int(u.sum())
    if mask_area < CHAN_VESE_MIN_AREA_RATIO * box_area_scaled:
        return None, None
    if mask_area > CHAN_VESE_MAX_AREA_RATIO * sh * sw:
        return None, None

    if scale != 1.0:
        u = cv2.resize(u, (cw, ch), interpolation=cv2.INTER_NEAREST)
    return u, (x0, y0)


def chan_vese_outline(image_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    """Obris boce iz Chan-Vese siluete (`_chan_vese_mask`) - koristi se
    kao zamjenski izvor obrisa kad `bottle_outline` (praćenje ruba s
    kontinuitetom) ne uspije, što je na stvarnom skupu bio slučaj za
    otprilike polovicu i inače ispravno lokaliziranih boca. Vraća poligon
    (N, 2) u apsolutnim koordinatama slike, ili None."""
    mask, origin = _chan_vese_mask(image_bgr, bbox)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, BOTTLE_SIMPLIFY_RATIO * perimeter, True)
    if len(simplified) < 3:
        return None
    return simplified.reshape(-1, 2) + np.array(origin, dtype=np.int32)


def chan_vese_silhouette(image_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    """Chan-Vese silueta poravnata s okvirom: binarna maska (h, w) u
    koordinatama `bbox`-a (izvan-okvirni dio siluete se odsijeca) ili
    None ako segmentacija ne uspije. Koristi je `chanvese_euklidska` za
    ograničenje pretrage razine tekućine na stvarnu unutrašnjost boce."""
    mask, origin = _chan_vese_mask(image_bgr, bbox)
    if mask is None:
        return None
    x, y, w, h = bbox
    ox, oy = origin
    out = np.zeros((h, w), dtype=np.uint8)
    src_x0 = max(0, x - ox); src_y0 = max(0, y - oy)
    src_x1 = min(mask.shape[1], x + w - ox); src_y1 = min(mask.shape[0], y + h - oy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return None
    dst_x0 = src_x0 + ox - x; dst_y0 = src_y0 + oy - y
    out[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = \
        mask[src_y0:src_y1, src_x0:src_x1]
    return out


REFINE_MIN_SIDE_RATIO = 0.55   # rafinirana stranica ne smije pasti ispod ovog udjela izvorne
REFINE_MAX_SIDE_RATIO = 1.45   # niti narasti preko ovog udjela - inače je segmentacija pobjegla
REFINE_MIN_IOU = 0.3           # rafinirani okvir mora ostati na istom objektu


def refine_boxes_chan_vese(image_bgr: np.ndarray, boxes: Sequence[BBox]) -> List[BBox]:
    """Pritezanje detektiranih okvira na stvarnu Chan-Vese siluetu boce -
    okviri iz kontura rubova/klastera znaju biti pomaknuti ili prelabavi
    (uhvate komad pozadine, odsijeku grlo), što IoU prema stvarnom okviru
    spušta ispod praga sparivanja iako je boca 'nađena'. Okvir se zamjeni
    okvirom siluete SAMO ako promjena ostane u razumnim granicama
    (`REFINE_*` - prevelika promjena znači da je segmentacija pobjegla u
    pozadinu ili kolabirala, pa je izvorni okvir pouzdaniji). Nakon
    pritezanja se ponovno uklone preklapanja - dva različita okvira znaju
    se stegnuti na ISTU bocu, a takav duplikat je prije bio lažni
    pozitiv."""
    refined: List[BBox] = []
    variance_map = None
    for box in boxes:
        mask, origin = _chan_vese_mask(image_bgr, box)
        if mask is None:
            refined.append(box)
            continue
        rx, ry, rw, rh = cv2.boundingRect(mask)
        candidate = (origin[0] + rx, origin[1] + ry, rw, rh)
        # Vraćanje vrha: kad pritezanje spusti vrh, vrati ga na najviši
        # redak s aktivnošću u lokalnoj varijanci unutar odrezane trake -
        # silueta zna izgubiti svijetli čep na svijetloj pozadini
        # (dijagnosticirano: crvena boca s bijelim čepom, pritezanje
        # odrezalo ~100 px čepa, IoU 0.81->0.94 s vraćanjem). Mjereno
        # punom evaluacijom (sa stats-window rezom): bez ovoga rgb gubi
        # jedan TP i 4 pb MAE, kmeans i chanvese ~0.5 pb, ansambl 1.3 pb;
        # jedino lab plaća ~0.2 pb - neto jasno pozitivno, zadržano.
        bx0, by0, _bw0, _bh0 = box
        cx0, cy0, cw0, ch0 = candidate
        if cy0 > by0:
            if variance_map is None:
                variance_map = local_variance_map(image_bgr)
            quarter = max(1, cw0 // 4)
            strip = variance_map[max(0, by0):cy0, cx0 + quarter:cx0 + cw0 - quarter]
            if strip.size:
                activity = (strip > 0).mean(axis=1)
                active = np.flatnonzero(activity >= EXTEND_TOP_ROW_ACTIVITY)
                if active.size:
                    new_top = max(0, by0) + int(active[0])
                    candidate = (cx0, new_top, cw0, ch0 + (cy0 - new_top))
        _, _, w, h = box
        if not (REFINE_MIN_SIDE_RATIO * w <= rw <= REFINE_MAX_SIDE_RATIO * w
                and REFINE_MIN_SIDE_RATIO * h <= rh <= REFINE_MAX_SIDE_RATIO * h):
            refined.append(box)
            continue
        if iou(candidate, box) < REFINE_MIN_IOU:
            refined.append(box)
            continue
        refined.append(candidate)
    # NAPOMENA: ovdje je RANIJE bio dodatni, bezuvjetan poziv
    # `extend_boxes_top(image_bgr, refined)` nakon petlje (obrazloženje:
    # silueta zna izgubiti čep/vrh boce, pa vrh treba ponovno provjeriti).
    # Ta zadaća je otad preuzeta preciznijim "vraćanjem vrha" UNUTAR petlje
    # iznad (ograničeno TOČNO na traku koju je pritezanje odrezalo, uz
    # IoU/omjer-stranica sigurnosne provjere) - bezuvjetan naknadni poziv
    # na VEĆ ispravljenim okvirima tražio je (nepostojeći) rub iznad
    # okvira koji je već točan, i znao naći nepovezanu teksturu pri vrhu
    # kadra (strop, okvir prozora) - dijagnosticirano: ispravan okvir
    # (IoU 0.97 s GT) proširen sve do y=0 (IoU pao na 0.85). Uklonjeno.
    return dedup_boxes(refined, max_boxes=len(refined) or 1)


def refine_boxes_to_outline(
    image_bgr: np.ndarray, boxes: Sequence[BBox], outline_edges: np.ndarray
) -> List[BBox]:
    """Gradijentno-VLASTITO pritezanje okvira (analogno
    `refine_boxes_chan_vese`, ali za lokalnu/gradijentnu paradigmu):
    svaki okvir se stegne na bounding-box VLASTITOG gradijentnog obrisa
    (`bottle_outline` - praćenje ruba nad kartom gradijentnih rubova), uz
    iste `REFINE_*` sigurnosne granice. Gradijentni algoritmi (lab/rgb)
    time NE posuđuju Chan-Vese siluetu za pritezanje - obris kojim se
    pritežu izlazi iz njihove vlastite paradigme (euklidska udaljenost
    susjednih piksela). Kad praćenje ruba ne uspije za neki okvir, taj se
    okvir ostavlja nedirnut. Na kraju se ukloni preklapanje (dva okvira
    znaju se stegnuti na istu bocu)."""
    refined: List[BBox] = []
    for box in boxes:
        outline = bottle_outline(box, outline_edges)
        if outline is None or len(outline) < 3:
            refined.append(box)
            continue
        rx, ry, rw, rh = cv2.boundingRect(outline.reshape(-1, 1, 2).astype(np.int32))
        candidate = (rx, ry, rw, rh)
        _, _, w, h = box
        if not (REFINE_MIN_SIDE_RATIO * w <= rw <= REFINE_MAX_SIDE_RATIO * w
                and REFINE_MIN_SIDE_RATIO * h <= rh <= REFINE_MAX_SIDE_RATIO * h):
            refined.append(box)
            continue
        if iou(candidate, box) < REFINE_MIN_IOU:
            refined.append(box)
            continue
        refined.append(candidate)
    return dedup_boxes(refined, max_boxes=len(refined) or 1)


_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
_KMEANS_SEED = 42  # fiksni seed - cv2.kmeans inače koristi dijeljeni globalni RNG,
                   # pa bi bez ovoga isti ulaz mogao dati različit rezultat ovisno
                   # o tome koliko je kmeans poziva već izvršeno u istom procesu.


def lab_saturation_features(image_bgr: np.ndarray) -> np.ndarray:
    """CIELAB + zasićenje (S) značajke po pikselu - dijeljena osnova za
    SVAKO K-means klasteriranje u projektu (razina tekućine/obris u
    `kmeans_euklidska`, fallback lokalizacije `color_cluster_candidate_boxes`
    ovdje) - isti prostor značajki, samo primijenjen na različite ulaze
    (cijela slika ili isječak oko okvira boce)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    saturation = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float64)
    return np.concatenate([lab, saturation[:, :, None]], axis=2)


def lab_saturation_spatial_features(image_bgr: np.ndarray, spatial_weight: float) -> np.ndarray:
    """CIELAB + zasićenje (S) + PONDERIRANE prostorne koordinate (x, y) po
    pikselu - prošireni prostor značajki za K-means LOKALIZACIJU u
    `kmeans_euklidska`. Dodavanje pozicije čini euklidsku udaljenost do
    centroida osjetljivom i na PROSTORNU udaljenost, pa se dvije boce ISTE
    boje na RAZLIČITIM mjestima (npr. red istovjetnih boca) razdvoje u
    zasebne klastere - inače bi kao jedna boja pale u isti klaster i spojile
    se u jednu (pre)široku regiju. Koordinate se normaliziraju na 0-255
    (raspon usporediv s Lab/S) i množe sa `spatial_weight` (koliko prostor
    "vuče" naspram boje; 0 = čisto bojno kao `lab_saturation_features`,
    veće = jače prostorno grupiranje - kalibrirano 0.3-0.8, 0.5 najbolji
    kompromis preciznost/odziv). I dalje ČISTI euklidski K-means, samo u
    prostoru boja+pozicije."""
    lab_s = lab_saturation_features(image_bgr)
    h, w = lab_s.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs = xs / max(1, w - 1) * 255.0 * spatial_weight
    ys = ys / max(1, h - 1) * 255.0 * spatial_weight
    return np.concatenate([lab_s, xs[:, :, None], ys[:, :, None]], axis=2)


def kmeans_cluster(feature_map: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dijeljeni K-means poziv (fiksni seed/kriteriji/broj pokušaja) -
    koriste ga i `kmeans_euklidska` (lokalizacija i razina tekućine) i
    `color_cluster_candidate_boxes` (fallback lokalizacije u ostalim
    algoritmima), da isti mehanizam klasteriranja ne bude dvaput
    implementiran. Vraća (oznake klastera oblika slike, centri klastera)."""
    h, w = feature_map.shape[:2]
    samples = feature_map.reshape(-1, feature_map.shape[-1]).astype(np.float32)
    cv2.setRNGSeed(_KMEANS_SEED)
    _, labels, centers = cv2.kmeans(
        samples, k, None, _KMEANS_CRITERIA, 4, cv2.KMEANS_PP_CENTERS
    )
    return labels.reshape(h, w), centers


CLUSTER_FALLBACK_K = 5    # isti broj klastera kalibriran za kmeans_euklidska.K_LOCALIZATION
CLUSTER_FALLBACK_MIN_TEXTURE = 0.02       # min. udio piksela s lokalnom varijancom unutar kandidata
CLUSTER_FALLBACK_MIN_SAT_EDGE = 0.08       # min. udio piksela sa signalom zasićenja unutar kandidata
CLUSTER_FALLBACK_MIN_AREA_RATIO = 0.07  # stroži prag SAMO za ovaj fallback (generalni
    # MIN_BOTTLE_AREA_RATIO=0.008 je namjerno permisivan za rub/klaster signale koji
    # daju malo lažnih kandidata - globalno bojno klasteriranje cijele slike zna
    # razdvojiti pozadinu (zid, sjenu, pod) u više manjih koherentnih fragmenata koji
    # isto prođu geometrijske filtre. Sve stvarne boce u aktivnom skupu (boce/, bez
    # boce/odvojeno) zauzimaju >=0.108 površine slike (izmjereno iz ground_truth.json)
    # - 0.07 sigurno ispod toga, a iznad izmjerenih lažnih kandidata (0.009-0.073)


def color_cluster_candidate_boxes(
    image_bgr: np.ndarray, existing: Sequence[BBox], k: int = CLUSTER_FALLBACK_K,
    iou_threshold: float = 0.25,
) -> List[BBox]:
    """Kandidati iz GRUBOG globalnog K-means klasteriranja (CIELAB+S) cijele
    slike - SAMO oni koje primarni (finiji) klaster-prolaz NIJE već pokrio.
    Koristi ga ISKLJUČIVO `kmeans_euklidska` kao dodatni, GRUBLJI (k=3)
    prolaz vlastite paradigme (nakon čišćenja paradigmi lab/rgb/chanvese ga
    više NE pozivaju - to bi za njih bila tuđa, particijska paradigma).
    Postoji jer providna boca s gradijentom osvjetljenja zna biti razlomljena
    preko više finih (K_LOCALIZATION) klastera, svaki premalen da prođe
    geometrijske filtre, dok grublje klasteriranje (manje klastera = veće,
    uključivije mrlje) zadrži cijelo tijelo boce u jednom komadu. Sigurnosni
    filtri niže (površina/tekstura/najveći preživjeli) sprječavaju da grublji
    prolaz doda krupne pozadinske mrlje (zid, pod)."""
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    h, w = blurred.shape[:2]
    labels, _centers = kmeans_cluster(lab_saturation_features(blurred), k)

    boxes: List[BBox] = []
    for ki in range(k):
        mask = (labels == ki).astype(np.uint8) * 255
        mask = exclude_skin(mask, image_bgr)
        boxes.extend(regions_from_binary_edges(mask))
    boxes = merge_and_dedup_boxes(boxes)
    img_area = h * w
    candidates = [
        b for b in boxes
        if b[2] * b[3] >= CLUSTER_FALLBACK_MIN_AREA_RATIO * img_area
        and all(iou(b, e) <= iou_threshold for e in existing)
    ]
    if not candidates:
        return candidates

    # Globalno klasteriranje zna razdvojiti pozadinu (zid, pod, sjena) u
    # velike, slučajno "bocolike" regije (visok, uzak oblik) koje prežive
    # geometrijske filtre i prag površine - dijagnosticirano: pozadinska
    # regija s IoU 0.09 prema stvarnoj boci, gotovo iste veličine kao
    # stvarna boca (prag površine/najveći-kandidat sam po sebi ih ne
    # razdvaja). Prava (čak i providna, gotovo nevidljiva) boca ipak IMA
    # neku unutarnju strukturu koje jednolična studijska pozadina/pod
    # nema - odsjaje, rub navoja čepa, granicu tekućine, nabore stakla -
    # mjereno: stvarna providna boca 5.6% pokrivenosti lokalnom varijancom
    # i 39.8% zasićenja-ruba, dok su sve dijagnosticirane pozadinske
    # regije bile <=0.5% odnosno <=0.4%. Zahtijeva se BILO KOJI od ova dva
    # neovisna signala (dovoljno velik razmak da jedan dovoljno pouzdano
    # razdvoji, drugi je dodatna sigurnost).
    variance_map = local_variance_map(image_bgr)
    sat_edge = exclude_skin(saturation_edge_map(image_bgr), image_bgr)
    plausible = []
    for b in candidates:
        bx, by, bw, bh = b
        var_cov = (variance_map[by:by + bh, bx:bx + bw] > 0).mean()
        sat_cov = (sat_edge[by:by + bh, bx:bx + bw] > 0).mean()
        if var_cov >= CLUSTER_FALLBACK_MIN_TEXTURE or sat_cov >= CLUSTER_FALLBACK_MIN_SAT_EDGE:
            plausible.append(b)

    # I nakon oba filtra (površina, tekstura) znaju preživjeti 2-3 velike
    # pozadinske regije istovremeno (npr. zid I pod, oboje s tragom
    # teksture od sjene/prozora) - u svim dijagnosticiranim slučajevima
    # gdje je ovaj fallback uopće trebao djelovati, stvarna (providna)
    # boca je bila NAJVEĆA preživjela regija. Zadržava se samo najveća kao
    # dodatna, jeftina provjera zdravog razuma - cijena je da bi u
    # (dosad neviđenom) prizoru s DVIJE stvarne, potpuno nevidljive boce
    # ovaj fallback uhvatio samo jednu; ta cijena je manja od dobiti
    # (manje lažnih pozitiva na daleko češćim jednobočnim scenama).
    if len(plausible) > 1:
        plausible = [max(plausible, key=lambda b: b[2] * b[3])]
    return plausible


EXTEND_TOP_MAX_RATIO = 0.35     # najviše ovoliko dodatne visine (udio postojeće visine okvira) smije se dodati
EXTEND_TOP_ROW_ACTIVITY = 0.12  # udio "teksturiranih" piksela u retku da se redak smatra dijelom boce
EXTEND_TOP_CONFIRM_GAP = 6      # broj uzastopnih "tihih" redaka prije prekida širenja (jedan šumni redak ne smije prekinuti)


def extend_boxes_top(image_bgr: np.ndarray, boxes: Sequence[BBox]) -> List[BBox]:
    """Proširuje okvire prema gore kad se čep/rame boce stopi s pozadinom
    (npr. bijeli čep na bijeloj pozadini) pa ga primarni rub/klaster signal
    uopće ne obuhvati - dijagnosticirano na stvarnom promašaju gdje je
    detektirani okvir izgubio gornjih ~270px boce (čep i rame), što je
    pomaklo bazu za računanje postotka punjenja i time samo po sebi
    proizvelo pogrešku od preko 25 postotnih bodova, neovisno o algoritmu
    razine tekućine (svi dijele lokalizaciju).

    Koristi `local_variance_map` (već postoji iz sličnog razloga - vidi
    njegov docstring): čak i blijed/proziran čep ima suptilnu unutarnju
    teksturu (odsjaje, rub navoja, utisnuti natpis) koju čista boja ili
    gradijent promaše, ali lokalna varijanca hvata. Širi se redak-po-redak
    od trenutnog vrha prema gore dok je udio "aktivnih" (teksturiranih)
    piksela u retku iznad `EXTEND_TOP_ROW_ACTIVITY`; prekida se tek nakon
    `EXTEND_TOP_CONFIRM_GAP` uzastopnih tihih redaka (ne kod prvog - jedan
    redak sjene/odsjaja ne smije prekinuti širenje), i najviše do
    `EXTEND_TOP_MAX_RATIO` * postojeća visina - sigurnosna granica koja
    sprječava bezgranično širenje u nepovezan objekt iznad boce (npr.
    policu, drugi predmet).

    NAPOMENA (testirano i odbačeno): zamjena globalnog Otsu praga (koji
    zna promašiti suptilan rub čepa kad natpis na etiketi NEGDJE DRUGDJE
    u kadru digne globalni prag previsoko) lokalnim Otsu pragom nad samom
    trakom pretrage, s biranjem "najjačeg" niza umjesto koraka-po-koraka.
    Popravilo je TOČNO dijagnosticirani slučaj (velika boca, IoU
    0.70->0.97), ali je punom evaluacijom potvrđeno NETO pogoršanje MAE
    (ansambl 7.0%->8.5%) - lokalni Otsu je nepouzdaniji na drugim slikama
    (zna izdvojiti nepovezanu teksturu - stropnu liniju, vinjetu vrha
    kadra - kao "jači" niz od stvarnog, ali suptilnijeg ruba boce).
    Vraćeno na globalni Otsu; ostaje otvoren problem za buduć rad."""
    variance_map = local_variance_map(image_bgr)
    h_img, w_img = variance_map.shape[:2]

    extended: List[BBox] = []
    for (x, y, w, h) in boxes:
        max_extend = int(h * EXTEND_TOP_MAX_RATIO)
        x0, x1 = max(0, x), min(w_img, x + w)
        new_top = y
        gap = 0
        for dy in range(1, max_extend + 1):
            row_y = y - dy
            if row_y < 0:
                break
            row = variance_map[row_y, x0:x1]
            if row.size == 0:
                break
            activity = float((row > 0).mean())
            if activity >= EXTEND_TOP_ROW_ACTIVITY:
                new_top = row_y
                gap = 0
            else:
                gap += 1
                if gap >= EXTEND_TOP_CONFIRM_GAP:
                    break
        if new_top < y:
            extended.append((x, new_top, w, y + h - new_top))
        else:
            extended.append((x, y, w, h))
    return extended


TEXTURE_SUPPLEMENT_MIN_AREA_RATIO = 0.045  # min. udio slike za varijancom-dopunjenu bocu (isto kao
                                            # kmeans texture fallback) - dovoljno velik da odbije šum,
                                            # ispod najmanje stvarne boce u skupu (0.108)


def texture_supplement_boxes(image_bgr: np.ndarray, boxes: Sequence[BBox]) -> List[BBox]:
    """Varijancom (lokalna tekstura) dobiveni bocoliki okviri koji NADILAZE
    (sadržajno) postojeće - dopuna za PROZIRNE boce (staklo/plastika) kojima
    primarni signal (gradijentni rub ILI Chan-Vese foreground) nađe samo
    SITAN komad koji ima kontrast (traka etikete, čep) dok cijelo bezbojno
    tijelo ostaje nevidljivo (IoU<0.4 pa boca broji i kao promašaj i kao
    lažni pozitiv - dijagnosticirano 175641/181228/175708). Lokalna varijanca
    je NEUTRALNA regijska/lokalna mjera (tekstura, ne posebna paradigma - već
    je fallback i u localize_gradient i u localize_variational), pa je
    dopuštena objema. Vraća SAMO okvire koji NISU već sadržani u nekom
    postojećem (`_contained_ratio < CONTAINMENT_DEDUP_RATIO`): na neprozirnim
    bocama je varijancin okvir ~jednak nađenom pa je sadržan i preskočen
    (nula promjene), a na sliver-promašaju puni okvir nadilazi sliver i doda
    se (naknadni containment-dedup odbaci sliver -> pobjeđuje potpuniji okvir).

    NAPOMENA: 175641 (prozirna boca u SLABOM svjetlu) ostaje nedostupna svima
    osim kmeansu - tamo tijelo NEMA NI teksture (ni Otsu ni dvoklasterska
    particija ne nalaze ništa osim slivera etikete), pa dopuna ne može pomoći;
    kmeans je nađe SAMO preko boje (klaster plave etikete). Čist primjer granice
    gradijentne/varijacijske paradigme koju particija premošćuje."""
    var_boxes = regions_from_binary_edges(exclude_skin(local_variance_map(image_bgr), image_bgr))
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    return [
        vb for vb in var_boxes
        if vb[2] * vb[3] >= TEXTURE_SUPPLEMENT_MIN_AREA_RATIO * img_area
        and all(_contained_ratio(vb, b) < CONTAINMENT_DEDUP_RATIO for b in boxes)
    ]


def localize_gradient(image_bgr: np.ndarray, gradient_fn) -> Tuple[List[BBox], np.ndarray]:
    """ČISTO GRADIJENTNA (lokalna/diferencijalna) lokalizacija - dijele je
    samo `lab_euklidska` i `rgb_euklidska` (razlikuju se po `gradient_fn`:
    magnituda gradijenta boje u CIELAB ili RGB). Ostaje ISKLJUČIVO u
    lokalnoj paradigmi (euklidska udaljenost susjednih piksela):

    1. primarni gradijent boje -> Otsu -> regije;
    2. fallback: fuzija s gradijentom ZASIĆENJA (||∇S||, i dalje gradijent)
       za mračne/niskokontrastne scene;
    3. zadnji fallback: lokalna varijanca (lokalna tekstura - ista
       lokalna/diferencijalna obitelj) kad boca gotovo nema kontrasta;
    4. proširenje vrha lokalnom teksturom (`extend_boxes_top`);
    5. pritezanje na VLASTITI gradijentni obris (`refine_boxes_to_outline`).

    NE posuđuje tuđe paradigme: nema udaljenosti-od-pozadine, K-means
    klasteriranja ni Chan-Vese pritezanja (odluka: čistoća paradigmi -
    prozirne boce koje bi uhvatio samo K-means fallback svjesno se
    prepuštaju, to je "ispravan smjer" za tvrdnju o 3 razdvojene paradigme).

    Vraća (boxes, binary): `binary` je binarna karta gradijentnih rubova
    (posljednja isprobana razina) - pozivatelji je ponovno koriste za obris
    (`compute_outline_edges` u rgb/lab_euklidska)."""
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    magnitude = gradient_fn(blurred)
    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = exclude_skin(binary, image_bgr)
    boxes = regions_from_binary_edges(binary)

    if not boxes:
        binary = exclude_skin(cv2.bitwise_or(binary, saturation_edge_map(blurred)), image_bgr)
        boxes = regions_from_binary_edges(binary)

    if not boxes:
        binary = exclude_skin(local_variance_map(image_bgr), image_bgr)
        boxes = regions_from_binary_edges(binary)

    # DOPUNA prozirnih boca (vidi texture_supplement_boxes): gradijentna
    # kaskada kod prozirne boce nađe samo sliver etikete/čepa, varijanca hvata
    # cijelo teksturirano tijelo.
    supplements = texture_supplement_boxes(image_bgr, boxes)
    if supplements:
        boxes = dedup_boxes(boxes + supplements, max_boxes=(len(boxes) + len(supplements)) or 1)

    boxes = extend_boxes_top(image_bgr, boxes)

    outline_edges = exclude_skin(cv2.bitwise_or(binary, saturation_edge_map(image_bgr)), image_bgr)
    boxes = refine_boxes_to_outline(image_bgr, boxes, outline_edges)
    return boxes, binary


CHAN_VESE_GLOBAL_MAX_DIM = 280     # radna rezolucija globalne ACWE segmentacije - dovoljna za
                                    # foreground boca, drži iteracije brzima
CHAN_VESE_GLOBAL_ITERATIONS = 60   # iteracija evolucije globalne konture


def global_chan_vese_foreground(image_bgr: np.ndarray) -> np.ndarray:
    """Globalna dvoregijska Chan-Vese (ACWE) segmentacija CIJELE slike na
    foreground (boce) / background: kontura evoluira tako da svaki piksel
    završi u regiji (unutar/izvan) čijem je PROSJEKU boje (CIELAB) euklidski
    bliži, uz morfološko zaglađivanje - minimizira sumu kvadratnih euklidskih
    odstupanja piksela od prosjeka vlastite regije. Vraća (H, W) masku
    foreground-a (0/255).

    Inicijalizacija iz udaljenosti od rubnih (pozadinskih) boja daje evoluciji
    razuman start (boce su obično dalje od boje pozadine nego pozadina od
    same sebe), a region-competition ga zatim popravi - hvata suptilne
    razlike koje bi tvrdi Otsu prag na toj udaljenosti promašio. Radi na
    umanjenoj slici radi brzine (nearest-upscale natrag na punu rezoluciju)."""
    H, W = image_bgr.shape[:2]
    scale = min(1.0, CHAN_VESE_GLOBAL_MAX_DIM / max(H, W))
    small = (cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1.0 else image_bgr)
    feats = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float64)
    sh, sw = feats.shape[:2]

    # Init: udaljenost od reprezentativnih boja ruba (pozadine), prag na medijan.
    band = max(2, int(0.04 * min(sh, sw)))
    refs = [
        np.median(feats[:band].reshape(-1, 3), axis=0),
        np.median(feats[-band:].reshape(-1, 3), axis=0),
        np.median(feats[:, :band].reshape(-1, 3), axis=0),
        np.median(feats[:, -band:].reshape(-1, 3), axis=0),
    ]
    dist = None
    for ref in refs:
        d = ((feats - ref) ** 2).sum(axis=2)
        dist = d if dist is None else np.minimum(dist, d)
    u = (dist > np.median(dist)).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(CHAN_VESE_GLOBAL_ITERATIONS):
        inside = u > 0
        n_in = int(inside.sum())
        if n_in < 10 or n_in > inside.size - 10:
            break
        c1 = feats[inside].mean(axis=0)
        c2 = feats[~inside].mean(axis=0)
        d1 = ((feats - c1) ** 2).sum(axis=2)
        d2 = ((feats - c2) ** 2).sum(axis=2)
        dil = cv2.dilate(u, kernel)
        ero = cv2.erode(u, kernel)
        boundary = dil != ero
        u[boundary & (d1 < d2)] = 1
        u[boundary & (d1 >= d2)] = 0
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, kernel)
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, kernel)

    if scale < 1.0:
        u = cv2.resize(u, (W, H), interpolation=cv2.INTER_NEAREST)
    return (u * 255).astype(np.uint8)


def regions_from_filled_mask(mask: np.ndarray, max_bottles: int = 6) -> List[BBox]:
    """Kandidati oblika boce iz PUNE (ispunjene) binarne maske - za razliku
    od `regions_from_binary_edges` (koja radi s TANKIM kartama rubova pa ih
    morfološki ZATVARA da spoji prekinute konture), ovdje se komponente vade
    IZRAVNO (`connectedComponents`, bez zatvaranja): u punoj foreground maski
    zatvaranje bi spojilo susjedne boce u jedan blob (dijagnosticirano na
    više-bočnim scenama - upravo je to prije gubilo cijele redove boca).
    Preširoke spojene regije razbija `_split_wide_box` (dolina stupčane
    gustoće između boca), a zatim slijede isti geometrijski filtri
    (površina/omjer/širina/rub) kao u `regions_from_binary_edges`."""
    h_img, w_img = mask.shape[:2]
    img_area = h_img * w_img
    binary = (mask > 0).astype(np.uint8)

    n_labels, _lab, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: List[BBox] = []
    for i in range(1, n_labels):
        x, y, w, h, area = (int(v) for v in stats[i])
        # `area` je stvaran broj piksela komponente (ne w*h bbox-a) - filtar
        # površine se odnosi na ispunjenost, kao i geometrijski filtri niže.
        if area < MIN_BOTTLE_AREA_RATIO * img_area or area > MAX_BOTTLE_AREA_RATIO * img_area:
            continue
        if x <= BORDER_MARGIN_PX or (x + w) >= w_img - BORDER_MARGIN_PX:
            continue
        aspect = h / float(w) if w else 0.0
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue
        if w > MAX_WIDTH_RATIO * w_img:
            continue
        boxes.append((x, y, w, h))

    # Sigurnosno razbijanje spojenih susjednih boca (dolina stupčane gustoće);
    # za već bocolike okvire je no-op (vidi `_split_wide_box`).
    split_boxes: List[BBox] = []
    for box in boxes:
        split_boxes.extend(_split_wide_box(box, binary, w_img))

    return dedup_boxes(split_boxes, max_boxes=max_bottles)


def localize_variational(image_bgr: np.ndarray) -> List[BBox]:
    """ČISTO VARIJACIJSKA (energetska) lokalizacija - koristi je samo
    `chanvese_euklidska`. Ostaje u varijacijskoj paradigmi (euklidska
    udaljenost piksela do STATISTIKE REGIJE):

    1. `global_chan_vese_foreground` = PRAVA dvoregijska Chan-Vese (ACWE)
       segmentacija CIJELE slike na foreground/background minimizacijom
       unutar-regijske varijance (sume kvadratnih euklidskih odstupanja od
       prosjeka svake regije) - dosljedno paradigmi, i za lokalizaciju, ne
       samo za obris/razinu;
    2. `regions_from_filled_mask` = povezane komponente foreground maske
       (BEZ morfološkog zatvaranja koje bi spojilo susjedne boce - vidi
       tamošnju napomenu), geometrijski filtrirane;
    3. `refine_boxes_chan_vese` = lokalna ACWE evolucija koja svaki okvir
       dodatno stegne na stvarnu siluetu.

    NE posuđuje gradijent (lokalna paradigma) ni K-means (globalna). RANIJA
    čista verzija koristila je obični Otsu prag na euklidskoj udaljenosti od
    boje pozadine kao primarni signal - zamijenjeno pravim globalnim ACWE-om:
    mjereno bitno bolje (IoU 0.66->0.74, MAE 19.1->12.6, TP 12->14) jer
    region-competition hvata suptilne razlike koje tvrdi prag promaši, a
    dosljednije je paradigmi (Chan-Vese energija i u lokalizaciji)."""
    mask = exclude_skin(global_chan_vese_foreground(image_bgr), image_bgr)
    boxes = regions_from_filled_mask(mask)
    if not boxes:
        # Krajnja nužda: i dalje regijski signal (varijanca kao mjera
        # unutar-regijske nehomogenosti), ne gradijent.
        binary = exclude_skin(local_variance_map(image_bgr), image_bgr)
        boxes = regions_from_binary_edges(binary)
    # DOPUNA prozirnih boca (vidi texture_supplement_boxes): globalni ACWE kod
    # prozirne boce boje pozadine zna dati samo djelomičan okvir (obojana
    # tekućina/etiketa), varijanca hvata cijelo teksturirano tijelo. Isti
    # regijski (ne-gradijentni) signal koji je već krajnja nužda gore.
    supplements = texture_supplement_boxes(image_bgr, boxes)
    if supplements:
        boxes = dedup_boxes(boxes + supplements, max_boxes=(len(boxes) + len(supplements)) or 1)
    # Fizikalno dovršavanje do vrha boce (čep/grlo): globalni foreground zna
    # stati na ramenu kad je bistri čep boje pozadine - `extend_boxes_top`
    # produži okvir dok ima teksture. Bijeli čep na bijeloj pozadini ostaje
    # slijepa točka (nema signala).
    boxes = extend_boxes_top(image_bgr, boxes)
    boxes = refine_boxes_chan_vese(image_bgr, boxes)
    return boxes


def _row_candidates(row_edges: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Sve pozicije jakog ruba u [lo, hi) dijelu retka - SVE, ne samo
    prva/najbliža - jer `_track_side` bira između njih onu najbližu
    trendu iz susjednih redaka, ne uvijek prvu od ruba (koja zna biti
    zavarana bližim artefaktom - odsjajem, otisnutim datumom na čepu -
    prije stvarnog zida boce)."""
    if hi <= lo:
        return np.array([], dtype=np.float64)
    return (np.flatnonzero(row_edges[lo:hi]) + lo).astype(np.float64)


def _track_side(candidates: List[np.ndarray], seed_row: int, seed_pos: float, max_jump: float) -> np.ndarray:
    """Prati jednu stranu (lijevu ili desnu) granice od `seed_row` prema
    gore i prema dolje: u svakom sljedećem retku bira se kandidat najbliži
    trenutnoj procjeni, prihvaćen samo ako je pomak unutar `max_jump` -
    stvaran zid boce mijenja poziciju postupno, pa nagli skok gotovo
    uvijek znači da je kandidat zapravo unutarnji artefakt, ne rub. Redci
    bez prihvatljivog kandidata ostaju NaN (popunjava ih interpolacija u
    `bottle_outline`), a trenutna procjena se ne mijenja dok se granica
    ponovno ne "uhvati"."""
    n = len(candidates)
    result = np.full(n, np.nan)
    result[seed_row] = seed_pos

    current = seed_pos
    for i in range(seed_row + 1, n):
        cands = candidates[i]
        if cands.size:
            idx = int(np.argmin(np.abs(cands - current)))
            if abs(cands[idx] - current) <= max_jump:
                current = cands[idx]
                result[i] = current

    current = seed_pos
    for i in range(seed_row - 1, -1, -1):
        cands = candidates[i]
        if cands.size:
            idx = int(np.argmin(np.abs(cands - current)))
            if abs(cands[idx] - current) <= max_jump:
                current = cands[idx]
                result[i] = current

    return result


def _moving_median(arr: np.ndarray, window: int) -> np.ndarray:
    """Pokretni medijan (neparni prozor) - uklanja preostali redak-po-redak
    šum iz praćenog profila granice bez nametanja unaprijed pretpostavljenog
    oblika (za razliku od uklapanja polinoma, prati i oštrije prijelaze
    poput ramena boce)."""
    if window < 3 or window % 2 == 0 or len(arr) < window:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    stacked = np.stack([padded[i:i + len(arr)] for i in range(window)])
    return np.median(stacked, axis=0)


def bottle_outline(bbox: BBox, binary_edges: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Stvaran obris boce, izveden izravno iz ruba na slici (ne unaprijed
    zadan predložak, ne grupiranje po boji). Umjesto da se u svakom retku
    uzme prvi rub od ruba okvira prema središtu (zna biti zavaran bližim
    artefaktom), granica se PRATI (`_track_side`): krene se od robusne
    (medijan preko srednje trake redaka) pozicije na sredini visine - tamo
    je signal obično najpouzdaniji - i redak-po-redak odabire kandidat
    najbliži dosadašnjem trendu, prema gore i prema dolje.

    Redci bez prihvatljivog kandidata (npr. posve providan dio boce na
    sličnoj pozadini) popunjavaju se interpolacijom iz susjednih pouzdanih
    redaka. Kod providnih boca je signal ipak i dalje šumovit, pa se
    profil na kraju zaglađuje širokim (postotak visine, ne fiksni broj
    piksela) pokretnim medijanom u nekoliko uzastopnih prolaza
    (`BOTTLE_SMOOTH_PASSES`). Medijan (ne polinom) je namjerno odabran jer
    čuva oštre prijelaze poput ramena boce (grlo/tijelo su otprilike
    ravni, prijelaz je nagao) - polinomsko uklapanje je isprobano i
    zaglađivalo je taj prijelaz u zaobljen "leć" oblik.

    Budući da profil po konstrukciji seže od vrha (y=0) do dna (y=h)
    okvira (interpolacijom/ekstrapolacijom preko cijele visine),
    `volumetric_fill_percent` ga smije koristiti bez dodatne provjere
    pokrivenosti visine.

    Ako `binary_edges` nije dan, ili nema dovoljno pouzdanog signala ni na
    sredini visine ni preko dovoljno redaka ukupno, vraća None - pozivni
    kod tada pada natrag na crtanje običnog pravokutnika `bbox`."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or binary_edges is None:
        return None

    h_img, w_img = binary_edges.shape[:2]
    x0, x1 = max(x, 0), min(x + w, w_img)
    y0, y1 = max(y, 0), min(y + h, h_img)
    crop = binary_edges[y0:y1, x0:x1]
    ch, cw = crop.shape
    if ch < 10 or cw < 10:
        return None

    margin = min(BOTTLE_EDGE_MARGIN_PX, cw // 4)
    center = cw // 2
    left_candidates = [_row_candidates(crop[i], margin, center) for i in range(ch)]
    right_candidates = [_row_candidates(crop[i], center, cw - margin) for i in range(ch)]

    seed_lo = max(0, int(ch * (0.5 - BOTTLE_SEED_BAND_RATIO / 2)))
    seed_hi = min(ch, int(ch * (0.5 + BOTTLE_SEED_BAND_RATIO / 2)))
    seed_row = (seed_lo + seed_hi) // 2
    left_seed_vals = [c[0] for c in left_candidates[seed_lo:seed_hi] if c.size]
    right_seed_vals = [c[-1] for c in right_candidates[seed_lo:seed_hi] if c.size]
    if len(left_seed_vals) < 2 or len(right_seed_vals) < 2:
        # Ni na sredini visine (obično najpouzdaniji dio) nema dovoljno
        # ruba za polazište praćenja - bolje bez obrisa nego izmisliti ga.
        return None

    max_jump = BOTTLE_TRACK_MAX_JUMP_RATIO * cw
    left = _track_side(left_candidates, seed_row, float(np.median(left_seed_vals)), max_jump)
    right = _track_side(right_candidates, seed_row, float(np.median(right_seed_vals)), max_jump)

    valid = ~np.isnan(left) & ~np.isnan(right)
    if valid.sum() < BOTTLE_MIN_ROW_COVERAGE * ch:
        return None

    rows = np.arange(ch, dtype=np.float64)
    left = np.interp(rows, rows[valid], left[valid])
    right = np.interp(rows, rows[valid], right[valid])

    window = max(BOTTLE_SMOOTH_WINDOW_MIN, int(ch * BOTTLE_SMOOTH_WINDOW_RATIO))
    if window % 2 == 0:
        window += 1
    for _ in range(BOTTLE_SMOOTH_PASSES):
        left = _moving_median(left, window)
        right = _moving_median(right, window)

    # Desna strana profila odozgo prema dolje, pa lijeva odozdo prema
    # gore - zatvara poligon bez potrebe za posebnim gornjim/donjim rubom.
    right_pts = list(zip(right.tolist(), rows.tolist()))
    left_pts = list(zip(left.tolist(), rows.tolist()))
    local_points = right_pts + list(reversed(left_pts))

    polygon = np.array(local_points, dtype=np.float64) + np.array([x0, y0], dtype=np.float64)
    polygon = polygon.astype(np.int32).reshape(-1, 1, 2)

    perimeter = cv2.arcLength(polygon, True)
    simplified = cv2.approxPolyDP(polygon, BOTTLE_SIMPLIFY_RATIO * perimeter, True)
    if len(simplified) < 3:
        return None

    return simplified.reshape(-1, 2)


def outline_width_profile(outline: Optional[np.ndarray], bbox: BBox) -> Optional[np.ndarray]:
    """Širina (u pikselima) stvarnog obrisa boce (`bottle_outline`) po
    retku unutar okvira - dijele je `volumetric_fill_percent` (presjek kao
    r(y)²) i `body_search_range` (gdje rame prelazi u tijelo, po tome što
    širina prestane rasti). None ako obris nije dostupan (pao natrag na
    pravokutnik) - pozivni kod tada ne dobiva oblikom-svjestan signal, ne
    izmišlja ga."""
    x, y, w, h = bbox
    if outline is None or len(outline) < 3 or h <= 0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = (outline - np.array([x, y], dtype=np.int32)).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [shifted], 255)
    return (mask > 0).sum(axis=1).astype(np.float64)


def volumetric_fill_percent(
    outline: Optional[np.ndarray], bbox: BBox, fill_percent_linear: float
) -> float:
    """Postotak VOLUMENA tekućine, ne samo linearnog udjela visine -
    mnoge boce nisu valjci (suženo grlo, bačvasto tijelo), pa kod njih
    linearna visina precjenjuje/podcjenjuje stvarni volumen. Boca se
    modelira kao rotacijsko tijelo: širina siluete u retku y tretira se
    kao promjer, pa je presjek ∝ r(y)² = (širina(y)/2)² (vidi
    V(h) = π∫r(y)²dy); volumni postotak je omjer sume presjeka ispod
    granice i ukupne sume.

    `fill_percent_linear` (već izračunat u `finalize_boundary`) koristi se
    da se odredi koji redak unutar okvira odgovara granici - izbjegava
    ponovno provlačenje `idx_median` kroz svaki modul.

    Ako `outline` nije dostupan (pao natrag na pravokutnik), širina je
    konstantna pa se rezultat svodi na isti broj kao `fill_percent_linear`
    - funkcija je tada bezopasan no-op, ne izmišlja podatak koji nema."""
    x, y, w, h = bbox
    if h <= 0:
        return fill_percent_linear

    width = outline_width_profile(outline, bbox)
    if width is None:
        return fill_percent_linear

    area_profile = width ** 2
    total = float(np.sum(area_profile))
    if total <= 0:
        return fill_percent_linear

    level_idx = int(np.clip(round(h * (1.0 - fill_percent_linear / 100.0)), 0, h))
    below = float(np.sum(area_profile[level_idx:]))
    return float(np.clip(100.0 * below / total, 0.0, 100.0))


BODY_WIDTH_TOLERANCE = 1.03        # udio referentne (tjelesne) širine od kojeg se rame smatra završenim - kalibrirano
                                    # grid-searchom (0.85-1.10); oštar prijelaz oko 1.03-1.05 (lab MAE skoči
                                    # 9.64->13.54) pa je ostavljena margina od ruba, ne najbolja pojedinačna točka
BODY_REFERENCE_BAND_RATIO = 0.35   # donji udio visine iz kojeg se procjenjuje "tjelesna" širina - kalibrirano
                                    # grid-searchom (0.15-0.40); prosjek MAE sva 4 algoritma pao s 10.98% (0.40) na 10.25%
BODY_CONFIRM_ROWS = 8              # broj uzastopnih redaka koji moraju potvrditi kraj ramena (ne 1 šumni redak)
SHOULDER_SAFETY_MARGIN_RATIO = 0.03  # dodatna margina nakon otkrivenog kraja ramena - ne rezati baš na rubu prijelaza - testirano 0.0-0.06, 0.03 ostaje najbolji
MAX_BODY_START_RATIO = 0.55        # gornja sigurnosna granica koliko visoko smijemo pomaknuti donju granicu pretrage - testirano 0.45-0.65, 0.55 ostaje najbolji


def body_search_range(
    outline: Optional[np.ndarray], bbox: BBox, search_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Prilagođava donju granicu raspona pretrage granice tekućine
    (`search_range[0]`, inače fiksni postotak visine jednak za sve oblike
    boca) stvarnom obliku boce. Rame/grlo ima jak, stvaran rub (nagla
    promjena širine i debljine stakla) koji euklidski gotovo uvijek
    nadjača stvaran, slabiji rub tekućine ako uopće uđe u raspon pretrage
    - dijagnosticirano na stvarnom promašaju gdje je gotovo prazna boca
    (tekućina pri dnu, ~18% ispunjenosti) prijavljena kao ~60% puna jer je
    `best_split_index` našao baš prijelaz rame->tijelo, ne meniskus.

    Koristi se širina stvarnog obrisa (`outline_width_profile`) po retku:
    rame završava tamo gdje širina prvi put (odozgo prema dolje) dosegne
    i ZADRŽI tjelesnu širinu (medijan donjih `BODY_REFERENCE_BAND_RATIO`
    redaka, gdje je gotovo sigurno već tijelo boce). Kraj ramena se traži
    kao prvi niz od `BODY_CONFIRM_ROWS` uzastopnih redaka širokih barem
    `BODY_WIDTH_TOLERANCE` puta tjelesne širine - jedan redak nije
    dovoljan (šum u obrisu), a `MAX_BODY_START_RATIO` sprječava da se kod
    pogrešne detekcije oblika donja granica pomakne toliko da gotovo ne
    ostavi prostora za pretragu.

    Isprobana je i grublja zamjena za slučaj kad `outline` nije dostupan
    (razmak krajnjeg lijevog/desnog rub-piksela po retku, bez zahtjeva za
    kontinuitetom praćenja) - na stvarnom ground truth uzorku pogoršala je
    MAE/RMSE kod sva tri algoritma (lažni "kraj ramena" iz udaljenog šuma
    pozadine u zagušenim scenama), pa je odbačena: kad obris nije
    dostupan, bolje bez oblik-signala nego s nepouzdanim.

    Ako `outline` nije dostupan (pao natrag na pravokutnik - konstantna
    širina) ili rame nije prepoznatljivo uže od tijela, vraća se
    `search_range` bez izmjene - bezopasan no-op kad oblik-signala nema."""
    lo, hi = search_range
    width = outline_width_profile(outline, bbox)
    if width is None:
        return search_range

    h = len(width)
    band_start = max(0, int(h * (1.0 - BODY_REFERENCE_BAND_RATIO)))
    reference = width[band_start:]
    if reference.size == 0:
        return search_range
    body_width = float(np.median(reference))
    if body_width <= 0:
        return search_range

    threshold = BODY_WIDTH_TOLERANCE * body_width
    wide_enough = width >= threshold

    shoulder_end = None
    run = 0
    for i, ok in enumerate(wide_enough):
        run = run + 1 if ok else 0
        if run >= BODY_CONFIRM_ROWS:
            shoulder_end = i - BODY_CONFIRM_ROWS + 1
            break
    if shoulder_end is None:
        return search_range

    shoulder_ratio = min(shoulder_end / h + SHOULDER_SAFETY_MARGIN_RATIO, MAX_BODY_START_RATIO)
    new_lo = max(lo, shoulder_ratio)
    if new_lo >= hi:
        return search_range
    return (new_lo, hi)


SPLIT_COL_VALLEY_RATIO = 0.35       # udio vrha stupčane gustoće ispod kojeg je stupac "dolina" (između boca)
SPLIT_SMOOTH_WINDOW_RATIO = 0.04    # prozor zaglađivanja stupčane gustoće (udio širine regije)
SPLIT_MIN_PART_WIDTH_RATIO = 0.08   # najuži dopušteni dio nakon rastavljanja (udio širine regije)
SPLIT_MAX_PARTS = 6                 # više dijelova od ovoga = vjerojatno tekstura pozadine, ne red boca
TIGHTEN_ROW_THRESHOLD_RATIO = 0.2   # udio vrha retčane gustoće ispod kojeg redak ne podupire dio


def _column_runs(active: np.ndarray) -> List[Tuple[int, int]]:
    """Uzastopni intervali True vrijednosti u 1D bool nizu, kao lista
    (početak, kraj-ekskluzivno) parova."""
    padded = np.concatenate(([False], active, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _split_wide_box(box: BBox, binary_edges: np.ndarray, w_img: int) -> List[BBox]:
    """Kad više boca stoji jedna uz drugu, njihovi se rubovi dodiruju pa
    se spoje u JEDNU široku konturu - takva regija ne nalikuje boci
    (preširoka / premalog omjera visina:širina) i filtri oblika bi je
    odbacili U CIJELOSTI, čime se gubi svih nekoliko boca odjednom
    (dijagnosticirano kao glavni uzrok niskog odziva na scenama s više
    boca). Umjesto odbacivanja, regija se pokuša rastaviti po "dolinama"
    stupčane gustoće ruba: unutar boce gustoća je visoka, u procjepu
    između dvije boce naglo padne. Svaki dovoljno širok "brijeg" postaje
    zaseban kandidat, vertikalno obrezan na retke koji stvarno imaju rub
    (boce u istom redu nisu nužno iste visine, a spojena kontura svima
    nameće zajednički okvir pune visine).

    Poziva se SAMO za regije koje filtre oblika ne bi prošle - kandidate
    koji već izgledaju kao boca ne dira, pa ne može pokvariti postojeće
    ispravne detekcije. Ako rastavljanje ne uspije (nema jasnih dolina)
    ili proizvede sumnjivo mnogo dijelova (`SPLIT_MAX_PARTS`, tipično
    tekstura pozadine poput fuga pločica, ne red boca), vraća se izvorna
    regija netaknuta."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return [box]
    if h / float(w) >= MIN_ASPECT_RATIO and w <= MAX_WIDTH_RATIO * w_img:
        return [box]

    crop = binary_edges[y:y + h, x:x + w] > 0
    col_density = crop.mean(axis=0)
    win = max(5, int(w * SPLIT_SMOOTH_WINDOW_RATIO)) | 1
    col_smooth = np.convolve(col_density, np.ones(win) / win, mode="same")
    peak = float(col_smooth.max())
    if peak <= 0:
        return [box]

    active = col_smooth > SPLIT_COL_VALLEY_RATIO * peak
    min_part_w = max(10, int(SPLIT_MIN_PART_WIDTH_RATIO * w))

    parts: List[BBox] = []
    for s, e in _column_runs(active):
        if e - s < min_part_w:
            continue
        rows = crop[:, s:e].mean(axis=1)
        rwin = max(5, int(h * 0.02)) | 1
        rows_smooth = np.convolve(rows, np.ones(rwin) / rwin, mode="same")
        top = float(rows_smooth.max())
        if top <= 0:
            continue
        supported = np.flatnonzero(rows_smooth > TIGHTEN_ROW_THRESHOLD_RATIO * top)
        r0, r1 = int(supported[0]), int(supported[-1])
        if r1 - r0 < 10:
            continue
        parts.append((x + int(s), y + r0, int(e - s), r1 - r0 + 1))

    if len(parts) <= 1 or len(parts) > SPLIT_MAX_PARTS:
        return [box]
    return parts


def regions_from_binary_edges(binary_edges: np.ndarray, max_bottles: int = 6) -> List[BBox]:
    """Pronalazi pravokutne regije koje po obliku (uski, izduženi, "puna"
    silueta) odgovaraju stojećim bocama, polazeći od već izračunate binarne
    karte (0/255) - kod RGB/CIELAB algoritama to je karta rubova, kod
    K-means algoritma maska jednog klastera. Ova funkcija samo generički
    obrađuje geometriju - svaki algoritam sam gradi svoju ulaznu kartu na
    svoj (euklidski) način, pa niti jedan korak ovdje nije specifičan za
    boju.

    Postupak u dvije faze:
    1. Ulazna karta se morfološki zatvori (spoji sitne prekide), a zatim se
       iz nje izvuku SVE konture (`RETR_LIST`, ne samo vanjske) - kod
       zagušenih/teksturiranih pozadina (npr. pločice) boce se često spoje
       s pozadinom u jednu veliku vanjsku konturu, dok pojedinačna boca
       ostaje vidljiva kao "mirna" unutarnja kontura (rupa) unutar te mase;
       kod K-means maske pojedinog klastera vanjska kontura samog klastera
       je kandidat.
    2. Sirovi kandidati se NAJPRIJE geometrijski spajaju (`_merge_stacked_boxes`)
       i tek se NAKON spajanja ocjenjuju stroža pravila oblika/površine.
       Da se filtriralo prije spajanja, fragment koji sam za sebe ne
       nalikuje boci (npr. samo gornja trećina tijela, prekinuta natpisom
       na etiketi) bio bi odbačen prije nego što dobije priliku spojiti se
       u cjelovitu konturu - upravo to je razlog zašto su boce s puno
       unutarnje teksture prije promašivane.
    """
    h_img, w_img = binary_edges.shape
    img_area = h_img * w_img

    closed = _close_binary_edges(binary_edges)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # "coverage" bilježi koliko je svakog piksela unutar konačnog okvira
    # stvarno prekriveno sirovim konturama koje su ušle u spajanje - to je
    # poopćenje starog "fill_ratio" (contourArea/bboxArea) mjerenog po
    # jednoj konturi, koje ostaje ispravno i nakon što se više kontura
    # spoji u jedan okvir.
    coverage = np.zeros((h_img, w_img), dtype=np.uint8)
    raw_boxes: List[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < MIN_FRAGMENT_AREA_RATIO * img_area or area > MAX_BOTTLE_AREA_RATIO * img_area:
            continue
        if x <= BORDER_MARGIN_PX or (x + w) >= w_img - BORDER_MARGIN_PX:
            continue
        raw_boxes.append((x, y, w, h))
        cv2.drawContours(coverage, [c], -1, 255, thickness=cv2.FILLED)

    merged = merge_stacked_boxes(raw_boxes)

    # Regije u kojima se više susjednih boca spojilo u jednu (široku)
    # konturu rastavljaju se PRIJE filtriranja po obliku - inače bi cijeli
    # red boca bio odbačen kao jedan "ne-boca" kandidat (vidi
    # `_split_wide_box`; ne dira regije koje već prolaze filtre oblika).
    split_boxes: List[BBox] = []
    for box in merged:
        split_boxes.extend(_split_wide_box(box, binary_edges, w_img))

    candidates: List[BBox] = []
    for (x, y, w, h) in split_boxes:
        area = w * h
        if area < MIN_BOTTLE_AREA_RATIO * img_area or area > MAX_BOTTLE_AREA_RATIO * img_area:
            continue
        aspect = h / float(w)
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue
        if w > MAX_WIDTH_RATIO * w_img:
            continue
        edge_density = (binary_edges[y:y + h, x:x + w] > 0).mean()
        if edge_density < MIN_EDGE_DENSITY:
            continue
        coverage_ratio = (coverage[y:y + h, x:x + w] > 0).mean()
        if coverage_ratio < MIN_COVERAGE_RATIO:
            continue
        candidates.append((x, y, w, h))

    return dedup_boxes(candidates, max_boxes=max_bottles)


# --------------------------------------------------------------------------
# Praćenje granice tekućine
# --------------------------------------------------------------------------
EDGE_BOOST_ALPHA = 6.0     # koliko jako lokalni gradijent pojačava euklidski rez - kalibrirano
                           # grid-searchom (0.2 do 8.0) izravno na GT okvirima; prosjek MAE
                           # sva 4 algoritma pao s 12.87% (izvorna vrijednost 0.4) na 11.97%
MENISCUS_CONTINUITY_ALPHA = 1.5  # ADITIVNI prior horizontalne kontinuiranosti (meniskus) - vidi
                           # best_split_index. 0 = isključeno (samo multiplikativni edge_boost).
                           # Za razliku od edge_boost-a (multiplikativan, ne može nadjačati puno
                           # jači bojni rez), aditivan je pa dovoljno kontinuiran vodoravan rub
                           # može preteći umjereno jaču bojnu razdvojenost - kalibrirati mjerenjem
SPLIT_STAT_CONTEXT_RATIO = 0.06  # koliko konteksta (udio visine okvira) iznad početka pretrage
                                 # ulazi u statistiku skupina - vidi komentar u best_split_index;
                                 # kalibrirano na GT okvirima (0.0-0.10 i "bez prozora"): 0.0 ruši
                                 # gotovo pune boce, bez prozora ruši gotovo prazne, 0.06 minimum
                                 # (prosjek MAE sva 4 algoritma 7.6% nasuprot 9.0% bez prozora)
CURVE_FIT_DEGREE = 1       # PRIOR VODORAVNOSTI (default za GRADIJENTNE lab/rgb; kmeans/chanvese
                           # prosljeđuju curve_degree=0, vidi boundary_curve). Stupanj 1 = LINIJA
                           # (dopušta blag perspektivni nagib), ne parabola. Za kmeans/chanvese (BEZ
                           # gradijentnog sidra, bučniji rezovi) čak i nagib lovi šum, pa oni koriste
                           # stupanj 0 (čista vodoravna - mjereno bolje: kmeans 10.13->9.62 na GT
                           # okvirima, i linija savršeno ravna/fizikalna). ISPROBANO i ODBAČENO za
                           # lab/rgb: stupanj 2 (plitka parabola) - na GT (savršenim) okvirima bolji
                           # (lab 8.90->8.77, rgb 6.52->6.33) jer hvata perspektivnu zakrivljenost,
                           # ALI na STVARNIM (labavim) detektiranim okvirima parabola se savije da
                           # uklopi grešku lokalizacije pa puni tijek i ansambl PORASTU - vraćeno na 1.
LEVEL_MAX_DEVIATION_RATIO = 0.05  # najveće dopušteno odstupanje razine od njenog medijana (udio visine
                           # okvira) - dio priora vodoravnosti: čak i nakon uklapanja linije, nijedna
                           # točka ne smije biti dalje od ovoga od vodoravne razine (sprječava ostatak
                           # "šatora"). Testirano 0.03-0.10; 0.05 dopušta realan meniskus/nagib, a reže
                           # ne-fizikalne skokove. Meniskus/perspektiva realno <= par % visine.
CURVE_FIT_MAD_K = 5.0      # prag (u MAD jedinicama) za odbacivanje odstupajućeg stupca - kalibrirano (1.5-10.0),
                           # rezultat konvergira iznad ~5 (odbacivanje odstupajućih stupaca sve rjeđe okine)


def best_split_index(
    values: np.ndarray,
    lo: int,
    hi: int,
    forbidden: Optional[np.ndarray] = None,
    edge_weight: Optional[np.ndarray] = None,
    continuity: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Umjesto najvećeg lokalnog skoka između susjednih redaka, traži
    globalno najbolji rez koji dijeli redke [0, h) na gornju i donju
    skupinu s maksimalnom RAZDVOJENOŠĆU srednjih vrijednosti skupina u
    euklidskom smislu - poopćenje na višedimenzijske (npr. RGB/Lab)
    vrijednosti. Znatno otpornije na pojedinačni šumni redak (odsjaj,
    mrlju) od lokalnog maksimuma razlike.

    Razdvojenost se mjeri Fisherovim (LDA) omjerom - kvadrat udaljenosti
    srednjih vrijednosti skupina (ponderiran veličinama skupina, `between`)
    podijeljen s UNUTAR-skupinskim rasprešenjem (`within`, suma kvadratnih
    odstupanja od srednje vrijednosti u svakoj skupini). Prijašnja verzija
    (klasični Otsu prag, `weight = n1*n2/h²`) mjerila je samo `between` -
    to sustavno favorizira rez blizu SREDINE raspona (n1*n2 je najveći kad
    su skupine jednake), jer visoko neuravnotežen rez dodatno kažnjava čak
    i kad je razlika boja ondje jasna. Za razinu tekućine to je pogrešna
    pretpostavka: kod gotovo prazne/gotovo pune boce stvaran rez je blizu
    ruba raspona, ne sredine. Dijeljenje s `within` tu pristranost uklanja
    - mala, homogena skupina (npr. 10% redaka iste boje) daje jednako
    visoku ocjenu kao velika, ako je razlika prema drugoj skupini stvarna
    i dosljedna (nizak within), umjesto da bude automatski kažnjena samo
    zbog svoje veličine.

    values: (h,) ili (h, k) niz. forbidden: bool maska (h,) - indeksi u
    kojima rez nije dopušten (npr. redci unutar etikete). edge_weight:
    (h,) niz - dodatna (neeuklidska) potvrda iz stvarnog gradijenta slike;
    pravu granicu tekućine gotovo uvijek prati i vidljiv rub (refleksija,
    menisk), pa se rez koji se poklapa s jakim gradijentom blago
    favorizira, čime je manje osjetljiv na "lažne" reze uzrokovane samo
    šumom boje.

    Povratna vrijednost je (split, pouzdanost). `split` je sub-pikselna
    (float): oko cjelobrojnog maksimuma provodi se parabolična
    interpolacija ocjene, što daje precizniju ocjenu razine nego
    zaokruživanje na najbliži redak. `pouzdanost` je omjer pobjedničke
    ocjene i MEDIJANA svih ocjena razmatranih u ovom pozivu - bezdimenzijski
    omjer (ne apsolutna vrijednost `score`-a, koja ovisi o mjernoj skali
    značajki - RGB 0-255, Lab, ili 0/1 pripadnost klasteru), pa isti prag
    pouzdanosti vrijedi za sve algoritme. Omjer blizu 1 znači da
    pobjednički rez nije ništa uvjerljiviji od tipičnog reza u ovom
    prozoru pretrage - dakle NEMA stvarne granice boje bilo gdje u
    dopuštenom rasponu (boca je providna bez signala, ili je gotovo posve
    prazna/puna pa je stvarna granica izvan raspona pretrage). Koristi ga
    pozivni kod (`boundary_curve`, `chanvese_euklidska.estimate_liquid_level`)
    da rezultat označi kao nepouzdan umjesto da tiho vrati samouvjeren,
    ali besmislen postotak - vidi CONFIDENCE_THRESHOLD i dijagnozu u
    evaluacija/rezultati (slika s tintanim staklom bez vidljive granice,
    gdje su sva četiri algoritma dosljedno pogodila krivo)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    h = values.shape[0]

    lo = max(lo, 1)
    hi = min(hi, h)
    if hi <= lo + 1:
        return float(lo), 0.0

    # Statistika skupina računa se SAMO nad prozorom [base, h), ne nad
    # svim redcima okvira: redci visoko iznad `lo` (čep, grlo, rame -
    # upravo ono što `body_search_range` isključuje iz pretrage) inače i
    # dalje glasaju u srednjim vrijednostima i varijancama skupina.
    # Dijagnosticirano na stvarnoj boci (tamno staklo, tekućina u donjih
    # 20%): gornja skupina ispravnog reza sadržavala je TRI vizualno
    # različite zone (čep + rame + prazno tijelo), pa joj je
    # unutar-varijanca eksplodirala i Fisherov omjer pao ispod omjera reza
    # na prijelazu rame->tijelo - iako je udaljenost srednjih boja na
    # ispravnom rezu bila jednako velika.
    #
    # `base` ipak NIJE točno `lo` nego malo iznad (SPLIT_STAT_CONTEXT_RATIO):
    # kod gotovo pune boce prava granica je odmah ispod `lo`, pa bi joj
    # gornja skupina bez konteksta imala svega nekoliko redaka - `between`
    # član (ponderiran veličinama skupina) tada propadne i pobijedi lažni
    # rez dublje u boci (izmjereno: boce s ~60-70% punjenja s -3 na -37
    # pb). Malen kontekst iznad `lo` (dio sigurnosne marže ramena, još
    # uvijek bez čepa/grla) daje rezovima uz vrh prozora dovoljnu gornju
    # skupinu, a zadržava glavni učinak isključenja gornjih zona.
    base = max(0, lo - int(SPLIT_STAT_CONTEXT_RATIO * h))
    win = values[base:]
    wn = win.shape[0]

    cumsum = np.cumsum(win, axis=0)
    total = cumsum[-1]
    cumsum_sq = np.cumsum(win ** 2, axis=0)
    total_sq = cumsum_sq[-1]

    idx = np.arange(lo + 1, hi)
    rel = idx - base
    top_sum = cumsum[rel - 1]
    top_sq = cumsum_sq[rel - 1]
    top_n = rel.astype(np.float64)
    bottom_n = wn - rel
    bottom_sum = total - top_sum
    bottom_sq = total_sq - top_sq

    top_mean = top_sum / top_n[:, None]
    bottom_mean = bottom_sum / bottom_n[:, None]
    dist = np.linalg.norm(top_mean - bottom_mean, axis=1)

    # Unutar-skupinsko rasprešenje (suma kvadratnih odstupanja od srednje
    # vrijednosti, po dimenziji, zbrojeno preko dimenzija - dosljedno
    # euklidskoj normi korištenoj za `dist`) - Fisherov nazivnik.
    within_top = np.sum(top_sq - (top_sum ** 2) / top_n[:, None], axis=1)
    within_bottom = np.sum(bottom_sq - (bottom_sum ** 2) / bottom_n[:, None], axis=1)
    within = np.clip(within_top + within_bottom, 0, None)

    between = (top_n * bottom_n / wn) * dist ** 2

    # Regularizacija razmjerna ukupnoj varijanci ovog prozora (ne fiksni
    # broj) - sprječava da omjer eksplodira kad je jedna skupina gotovo
    # savršeno homogena (within blizu nule) samo zbog svoje male veličine,
    # bez stvarnog signala.
    total_variance = float(np.sum(total_sq - (total ** 2) / wn))
    eps = max(total_variance, 1.0) * 1e-4
    score = between / (within + eps)

    if edge_weight is not None:
        w_slice = np.asarray(edge_weight, dtype=np.float64)[idx]
        w_norm = w_slice / (w_slice.max() + 1e-9)
        score = score * (1.0 + EDGE_BOOST_ALPHA * w_norm)

    # ADITIVNI PRIOR HORIZONTALNE KONTINUIRANOSTI (meniskus): multiplikativni
    # `edge_weight` iznad ne može nadjačati rez na kojem je BOJNA razdvojenost
    # (Fisher) puno veća nego na pravom meniskusu (npr. dno boce, rub etikete,
    # boja stakla kod prozirnih/tamnih boca - dijagnosticirano: lab/kmeans/
    # chanvese ondje zaključaju rez, dok rgb pogodi). `continuity` je GLOBALNI
    # profil (0-1) koliko je vodoravni rub u tom retku KONTINUIRAN preko cijele
    # širine boce - površina tekućine je vodoravna linija, pa pravi meniskus
    # ima visoku kontinuiranost, a lokalni artefakt (nabor, odsjaj) nisku.
    # Dodaje se PROPORCIONALNO tipičnoj ocjeni ovog prozora (medijan pozitivnih),
    # pa dovoljno kontinuiran meniskus može preteći umjereno jači bojni rez, ali
    # ne može izmisliti granicu ondje gdje kontinuiteta nema (c≈0 -> nema dodatka).
    if continuity is not None:
        c_slice = np.asarray(continuity, dtype=np.float64)[idx]
        c_norm = c_slice / (c_slice.max() + 1e-9)
        pos_scores = score[score >= 0]
        typ = float(np.median(pos_scores)) if pos_scores.size else 0.0
        score = score + MENISCUS_CONTINUITY_ALPHA * c_norm * typ

    if forbidden is not None:
        score = np.where(forbidden[idx], -1.0, score)

    if np.all(score < 0):
        return float(idx[len(idx) // 2]), 0.0

    pos = int(np.argmax(score))
    split = float(idx[pos])

    # NAPOMENA: regularizacija ovdje NAMJERNO nije `eps` iznad (koji je
    # skaliran prema varijanci ULAZNIH značajki - RGB 0-255, Lab, ili
    # 0/1 pripadnost klasteru) - `score` je već bezdimenzijski omjer
    # (between/within), reda veličine ~1, pa bi isti `eps` (koji za RGB
    # zna biti reda veličine 1 ili više) gurnuo pouzdanost prema nuli
    # BEZ OBZIRA na stvarni signal (dijagnosticirano: RGB/Lab su svi
    # ispali "nepouzdani" dok je K-means, s malenom varijancom 0/1
    # značajki pa i malenim `eps`, ispravno pokazivao raspon). Ovdje
    # treba sitna, apsolutna konstanta - samo da spriječi dijeljenje s
    # nulom kad je `baseline` doslovno 0.
    valid_scores = score[score >= 0]
    baseline = float(np.median(valid_scores)) if valid_scores.size else 0.0
    confidence = float(score[pos] / (baseline + 1e-9))

    # Sub-pikselna (parabolična) korekcija oko diskretnog maksimuma.
    if 0 < pos < len(score) - 1 and score[pos - 1] >= 0 and score[pos + 1] >= 0:
        denom = score[pos - 1] - 2 * score[pos] + score[pos + 1]
        if abs(denom) > 1e-9:
            delta = 0.5 * (score[pos - 1] - score[pos + 1]) / denom
            split += float(np.clip(delta, -1.0, 1.0))

    return split, confidence


LABEL_COLOR_BAND_FACTOR = 1.8  # MAD-prag za varijancu boje po retku - krupni grafički blokovi (logo/značka) - testirano 1.0-3.0, 1.8 ostaje najbolji


def label_band_mask(
    gray_strip: np.ndarray,
    factor: float = 1.2,  # kalibrirano (0.8-1.8), 1.2 blago bolji od izvornih 1.8
    dilate: int = 5,  # testirano 3-9, 5 ostaje najbolji
    feature_strip: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Redci s tekstom etikete imaju znatno veći horizontalni gradijent
    od glatke tekućine/zraka/stakla - ta razlika koristi se da se isključe
    iz kandidata za granicu tekućine (gdje inače najčešće uzrokuju lažni
    "najveći skok").

    Gusta tekstura teksta nije jedini izvor lažnih rezova: KRUPNI
    grafički blokovi etikete (logo, značka, ikona) mogu imati posve
    glatku unutrašnjost (nizak horizontalni gradijent teksta), ali
    dosljedno DRUGAČIJU boju od susjednog dijela etikete/stakla -
    dijagnosticirano na stvarnom promašaju gdje su sva četiri algoritma
    (RGB/Lab/K-means/Chan-Vese) rub loga/značke pogrešno uzela za granicu
    tekućine iako je cijela boca bila puna. Ako je `feature_strip` (h, w,
    k) dan (isti isječak značajki - RGB/Lab - koji `boundary_curve` već
    ima za ovaj stupac), dodatno se traže redci gdje se STUPČANA
    (unutar-retka) raspršenost boje naglo promijeni u odnosu na susjedne
    retke - to hvata RUB grafičkog bloka (gdje boja skokovito prijeđe iz
    jedne u drugu regiju) bez potrebe za teksturom teksta."""
    if gray_strip.shape[1] < 2:
        return np.zeros(gray_strip.shape[0], dtype=bool)
    grad = cv2.Sobel(gray_strip.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    row_texture = np.mean(np.abs(grad), axis=1)
    med = np.median(row_texture)
    mad = np.median(np.abs(row_texture - med)) + 1e-6
    mask = row_texture > (med + factor * mad)

    if feature_strip is not None and feature_strip.shape[1] >= 2:
        row_std = feature_strip.astype(np.float64).std(axis=1)   # (h, k)
        row_spread = np.linalg.norm(row_std, axis=1)              # (h,)
        # Promjena raspršenosti boje redak-na-redak (ne apsolutna
        # vrijednost, koja je visoka i unutar same tekućine kod
        # teksturiranog stakla) - rub grafičkog bloka izgleda kao nagli
        # skok u ovoj mjeri, dosljedno kroz kanale boje.
        spread_jump = np.abs(np.diff(row_spread, prepend=row_spread[0]))
        med2 = np.median(spread_jump)
        mad2 = np.median(np.abs(spread_jump - med2)) + 1e-6
        color_mask = spread_jump > (med2 + LABEL_COLOR_BAND_FACTOR * mad2)
        mask = mask | color_mask

    if dilate > 1:
        mask = np.convolve(mask.astype(np.uint8), np.ones(dilate), mode="same") > 0
    return mask


def _robust_curve_fit(xs: np.ndarray, ys: np.ndarray, degree: int, mad_k: float) -> np.ndarray:
    """Prilagođava (nisko-stupanjski) polinom točkama granice i jednom
    iteracijom odbacuje odstupajuće stupce (npr. onaj u kojem je odsjaj
    zavarao euklidski rez) prije konačnog uklapanja - poopćenje RANSAC-a
    prilagođeno malom broju točaka (`N_SEGMENTS`)."""
    coeffs = np.polyfit(xs, ys, degree)
    fitted = np.polyval(coeffs, xs)
    residual = ys - fitted
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-6
    inliers = np.abs(residual) <= mad_k * mad
    if inliers.sum() >= degree + 2 and not inliers.all():
        coeffs = np.polyfit(xs[inliers], ys[inliers], degree)
        fitted = np.polyval(coeffs, xs)
    return fitted


def finalize_boundary(points: List[Point], bbox: BBox) -> Tuple[List[Point], float]:
    """Dovršava rezultat granice tekućine za jednu bocu: median stupčanih
    rezova daje jedinstveni postotak punjenja, a točke krivulje se pomiču
    iz koordinata okvira u apsolutne koordinate slike. Identičan zadnji
    korak u sva tri algoritma (`estimate_liquid_level`), pa je ovdje
    zajednički."""
    x, y, w, h = bbox
    if not points:
        points = [(w / 2.0, h * 0.5)]

    idx_median = float(np.median([p[1] for p in points]))
    fill_percent = 100.0 * (h - idx_median) / h
    curve_abs = [(x + px, y + py) for px, py in points]
    return curve_abs, float(np.clip(fill_percent, 0, 100))


def boundary_curve(
    feature_map: np.ndarray,
    gray_roi: np.ndarray,
    margin_ratio: float,
    search_range: Tuple[float, float],
    n_segments: int = 9,
    mask: Optional[np.ndarray] = None,
    use_edge_boost: bool = True,
    use_label_band: bool = True,
    curve_degree: Optional[int] = None,
) -> Tuple[List[Point], float]:
    """Boca je 3D objekt pa granica tekućine na fotografiji općenito nije
    ravna crta preko cijele širine, nego blago zaobljena linija (perspektiva
    + zakrivljenost stakla/plastike). Umjesto jedne granice za cijelu širinu
    okvira, širina se dijeli na `n_segments` stupaca i granica se traži
    neovisno u svakom stupcu, čime se dobiva stvarni (potencijalno
    zakrivljeni) profil umjesto pretpostavljene ravne linije.

    feature_map: (h, w, k) niz značajki (RGB/Lab vrijednosti ili udio
    klastera) za cijeli okvir boce. `mask`: opcionalna (h, w) bool maska
    (npr. Chan-Vese silueta - vidi chanvese_euklidska.estimate_liquid_level)
    - kad je dana, prosjek po retku unutar svakog stupca računa se SAMO
    nad pikselima gdje je maska istinita, ne nad cijelom širinom stupca,
    pa pozadina unutar pravokutnog okvira ne glasa za granicu. Redci bez
    ijednog maskiranog piksela u stupcu popunjavaju se interpolacijom iz
    susjednih redaka istog stupca.

    `use_edge_boost`/`use_label_band`: pojačanje reza stvarnim vertikalnim
    gradijentom (menisk kao vidljiv rub) i isključivanje redaka etikete po
    horizontalnom gradijentu teksta - OBOJE su GRADIJENTNI (lokalna/
    diferencijalna paradigma), pa su uključeni SAMO za gradijentne
    algoritme (lab/rgb, kojima je gradijent vlastita paradigma). K-means
    (particijska) i Chan-Vese (varijacijska) paradigma NE smiju posuđivati
    gradijent - njihova procjena razine ostaje čisto u vlastitoj paradigmi
    (udio klastera / prosjek regije unutar siluete), pa im se ovi pomoćnici
    isključuju (odluka: čistoća paradigmi ispred MAE-a).

    Vraća (points, pouzdanost): `points` je lista (x, y) sub-pikselnih
    točaka granice u koordinatama okvira (relativno na gornji lijevi kut
    bbox-a); `pouzdanost` je MEDIJAN pouzdanosti (`best_split_index`) preko
    svih stupaca - jedan šumni stupac ne smije sam odrediti je li cijela
    granica pouzdana."""
    h, w = feature_map.shape[:2]
    margin = int(w * margin_ratio)
    core_w = max(1, w - 2 * margin)
    seg_width = max(1, core_w // n_segments)

    lo = int(h * search_range[0])
    hi = max(int(h * search_range[1]), lo + 1)

    # Vertikalni gradijent (Sobel po y): prava granica tekućine gotovo
    # uvijek prati i stvaran vidljiv rub (menisk, refleksija), ne samo
    # promjenu boje - ovaj profil pojačava (ne zamjenjuje) euklidski rez.
    # Računa se samo kad je dopušten (gradijentni algoritmi).
    grad_y = (
        np.abs(cv2.Sobel(gray_roi.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3))
        if use_edge_boost else None
    )
    # Globalni profil HORIZONTALNE KONTINUIRANOSTI ruba: svaki stupac se
    # normalizira svojim maksimumom (par vrlo svijetlih stupaca ne dominira),
    # pa se uzme prosjek po stupcima jezgre - redak s vodoravnim rubom preko
    # cijele širine dobije visoku vrijednost, lokalni artefakt (jak u par
    # stupaca) nisku. Isti profil ide u SVAKI stupac kao aditivni prior
    # meniskusa (vidi best_split_index). Samo kad je gradijent dopušten.
    if grad_y is not None:
        core_grad = grad_y[:, margin:margin + core_w]
        if core_grad.shape[1] >= 1:
            continuity = (core_grad / (core_grad.max(axis=0, keepdims=True) + 1e-9)).mean(axis=1)
        else:
            continuity = None
    else:
        continuity = None

    rows_idx = np.arange(h, dtype=np.float64)
    points: List[Point] = []
    confidences: List[float] = []
    for s in range(n_segments):
        x0 = margin + s * seg_width
        x1 = margin + core_w if s == n_segments - 1 else margin + (s + 1) * seg_width
        if x1 - x0 < 1:
            continue
        strip = feature_map[:, x0:x1]
        if mask is not None:
            mask_strip = mask[:, x0:x1]
            counts = mask_strip.sum(axis=1).astype(np.float64)
            sums = (strip * mask_strip[..., None]).sum(axis=1)
            row_vals = np.divide(
                sums, counts[:, None], out=np.full((h, strip.shape[-1]), np.nan),
                where=counts[:, None] > 0,
            )
            valid = counts > 0
            if valid.sum() >= 2:
                for c in range(row_vals.shape[1]):
                    row_vals[:, c] = np.interp(rows_idx, rows_idx[valid], row_vals[valid, c])
            else:
                row_vals = np.median(strip.reshape(h, -1, strip.shape[-1]), axis=1)
        else:
            row_vals = np.median(strip.reshape(h, -1, strip.shape[-1]), axis=1)
        if use_label_band:
            forbidden = label_band_mask(gray_roi[:, x0:x1], feature_strip=strip)
            # Ako natpis na etiketi (ili druga tekstura) pokrije gotovo cijeli
            # dopušteni raspon pretrage u ovom stupcu, zabrana se zanemaruje -
            # bolje pretražiti cijeli raspon nego rez prisilno stisnuti u uski
            # preostatak (koji tada gotovo nasumično određuje granicu).
            if forbidden[lo:hi].mean() > 0.85:
                forbidden = None
        else:
            forbidden = None
        edge_profile = grad_y[:, x0:x1].mean(axis=1) if grad_y is not None else None
        split, confidence = best_split_index(
            row_vals, lo, hi, forbidden, edge_weight=edge_profile, continuity=continuity
        )
        points.append((x0 + (x1 - x0) / 2.0, split))
        confidences.append(confidence)

    # PRIOR VODORAVNOSTI: svaki stupac traži rez neovisno, pa uz bučan signal
    # (etiketa, odsjaj, embosirani tekst na staklu) profil zna odlutati u
    # ne-fizikalan oblik. U stvarnosti je površina tekućine VODORAVNA (uz
    # blagi meniskus/perspektivu), pa se profilu robusno prilagođava skoro
    # vodoravna LINIJA (`CURVE_FIT_DEGREE=1`) uz odbacivanje odstupajućih
    # stupaca, a odstupanja od medijana se dodatno OGRANIČE na malen udio
    # visine (`LEVEL_MAX_DEVIATION_RATIO`) - time nijedan preostali šum ne
    # može proizvesti "šator/luk". Fizikalno znanje o tekućini (razina je
    # vodoravna), ne mjerni trik.
    # Stupanj uklapanja: default CURVE_FIT_DEGREE=1 (linija - dopušta blag
    # perspektivni nagib) za gradijentne (lab/rgb), gdje su per-stupac rezovi
    # ČISTI (usidreni gradijentom). Čiste particijska/varijacijska paradigma
    # (kmeans/chanvese, BEZ gradijentnog sidrenja) daju bučnije rezove pa im
    # nagib samo lovi šum - one prosljeđuju curve_degree=0 (čista vodoravna
    # linija = robustni prosjek stupaca), što je i točnije (mjereno: kmeans
    # 10.13->9.62, chanvese 8.93->8.89) i fizikalno čišće (linija savršeno
    # vodoravna, bez nagiba od šuma).
    deg = CURVE_FIT_DEGREE if curve_degree is None else curve_degree
    if len(points) >= 3:
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        level = float(np.median(ys))
        if len(points) >= deg + 2:
            ys_fit = _robust_curve_fit(xs, ys, deg, CURVE_FIT_MAD_K)
        else:
            ys_fit = np.full_like(ys, level)
        max_dev = LEVEL_MAX_DEVIATION_RATIO * h
        ys_fit = np.clip(ys_fit, level - max_dev, level + max_dev)
        points = list(zip(xs.tolist(), ys_fit.tolist()))

    overall_confidence = float(np.median(confidences)) if confidences else 0.0
    return points, overall_confidence


# --------------------------------------------------------------------------
# Vizualizacija
# --------------------------------------------------------------------------
BOX_COLOR = (60, 220, 60)
CURVE_COLOR = (0, 60, 255)
LIQUID_WASH_COLOR = (255, 170, 40)
LIQUID_WASH_ALPHA = 0.25
GAUGE_WIDTH = 16
GAUGE_MARGIN = 10


def _gauge_color(fill_percent: float) -> Tuple[int, int, int]:
    """Boja mjerača tekućine (crvena -> žuta -> zelena) - BGR."""
    t = float(np.clip(fill_percent / 100.0, 0.0, 1.0))
    red, amber, green = np.array([40, 40, 220]), np.array([0, 200, 255]), np.array([90, 190, 40])
    if t < 0.5:
        color = red + (amber - red) * (t / 0.5)
    else:
        color = amber + (green - amber) * ((t - 0.5) / 0.5)
    return tuple(int(c) for c in color)


def _draw_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.65,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
    thickness: int = 2,
) -> None:
    """Iscrtava tekst s polu-prozirnom pozadinom radi čitljivosti na
    svjetlim i tamnim dijelovima fotografije (umjesto dvostrukog obrisa
    crno/bijelo)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(int(org[0]), 0)
    y = max(int(org[1]), th + 4)
    pad = 5
    x0, y0 = max(x - pad, 0), max(y - th - pad, 0)
    x1, y1 = min(x + tw + pad, img.shape[1] - 1), min(y + baseline + pad, img.shape[0] - 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.putText(img, text, (x, y), font, scale, fg, thickness, cv2.LINE_AA)


def _draw_liquid_wash(
    img: np.ndarray, bbox: BBox, curve_points: Sequence[Point], outline: Optional[np.ndarray] = None
) -> None:
    """Polu-prozirna "vodena" ispuna ispod granice tekućine - razinu čini
    trenutno vidljivom na prvi pogled, ne samo kao brojku ili tanku liniju.
    Ispuna se siječe sa stvarnim obrisom boce (`outline`), ako je pronađen,
    da ne "curi" izvan boce kad okvir (`bbox`) nije tijesno uz tijelo (npr.
    nagnuta boca ili susjedna boca unutar istog pravokutnika)."""
    if len(curve_points) < 2:
        return
    x, y, w, h = bbox
    top_y = curve_points[0][1]
    bottom_y = curve_points[-1][1]
    poly = [(x, top_y)] + list(curve_points) + [(x + w, bottom_y), (x + w, y + h), (x, y + h)]
    poly_arr = np.array(poly, dtype=np.int32)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_arr], 255)

    if outline is not None and len(outline) >= 3:
        shape_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(shape_mask, [outline.reshape(-1, 1, 2).astype(np.int32)], 255)
        mask = cv2.bitwise_and(mask, shape_mask)

    overlay = img.copy()
    overlay[mask.astype(bool)] = LIQUID_WASH_COLOR
    cv2.addWeighted(overlay, LIQUID_WASH_ALPHA, img, 1 - LIQUID_WASH_ALPHA, 0, dst=img)


def _draw_fill_gauge(img: np.ndarray, bbox: BBox, fill_percent: float) -> None:
    """Uski okomiti mjerač uz okvir boce (kao baterija) - razinu tekućine
    čini vidljivom neovisno o krivulji/postotku, korisno za brzu vizualnu
    usporedbu više boca na istoj fotografiji."""
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]

    gx0 = x + w + GAUGE_MARGIN
    if gx0 + GAUGE_WIDTH >= img_w:
        gx0 = max(0, x - GAUGE_MARGIN - GAUGE_WIDTH)
    gx1 = min(gx0 + GAUGE_WIDTH, img_w - 1)

    cv2.rectangle(img, (gx0, y), (gx1, y + h), (255, 255, 255), 1, cv2.LINE_AA)
    fill_h = int(round(h * np.clip(fill_percent, 0, 100) / 100.0))
    if fill_h > 2:
        fy0 = y + h - fill_h
        cv2.rectangle(img, (gx0 + 2, fy0), (gx1 - 2, y + h - 2), _gauge_color(fill_percent), cv2.FILLED)


def _draw_bottle_shape(img: np.ndarray, bbox: BBox, outline: Optional[np.ndarray]) -> None:
    """Ocrtava stvarni obris boce (`bottle_outline`) ako je pronađen,
    kontrastno kao i krivulja granice tekućine (bijeli oreol + boja);
    inače pada natrag na obični pravokutnik okvira."""
    if outline is not None and len(outline) >= 3:
        pts = outline.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], isClosed=True, color=BOX_COLOR, thickness=2, lineType=cv2.LINE_AA)
    else:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2, cv2.LINE_AA)


def annotate_bottles(image_bgr: np.ndarray, results: Sequence[BottleResult], algorithm_label: str) -> np.ndarray:
    """results: lista BottleResult (bbox, curve_points_abs, fill_percent,
    outline, fill_percent_volume, confidence), gdje je curve_points_abs
    lista (x, y) točaka granice tekućine u apsolutnim koordinatama slike.
    Za svaku boci se iscrtavaju: stvarni obris (ili okvir kao zamjena),
    polu-prozirna ispuna razine tekućine, kontrastna (bijelim oreolom
    uokvirena) krivulja granice, okomiti mjerač razine te čitljiva oznaka
    postotka (uz volumni postotak kao manju napomenu, ako se od linearnog
    razlikuje za >1 p.b.). Kad je `confidence` ispod
    `CONFIDENCE_THRESHOLD` (nema stvarne granice boje u rasponu pretrage -
    vidi `best_split_index`), oznaka dobiva "[?]" i žutu pozadinu umjesto
    tamne - vizualno upozorenje da je taj postotak nagađanje, ne mjerenje,
    korisno za brzu vizualnu provjeru pri pokretanju main.py."""
    out = image_bgr.copy()
    for bbox, curve_points, fill_percent, outline, fill_percent_volume, confidence in results:
        x, y, w, h = bbox

        _draw_liquid_wash(out, bbox, curve_points, outline)
        _draw_bottle_shape(out, bbox, outline)

        if len(curve_points) >= 2:
            pts = np.array(curve_points, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
            cv2.polylines(out, [pts], isClosed=False, color=CURVE_COLOR, thickness=2, lineType=cv2.LINE_AA)
            for px, py in curve_points:
                cv2.circle(out, (int(px), int(py)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (int(px), int(py)), 2, CURVE_COLOR, -1, cv2.LINE_AA)

        _draw_fill_gauge(out, bbox, fill_percent)

        low_confidence = confidence < CONFIDENCE_THRESHOLD
        label_text = f"{fill_percent:.0f}%" + ("  [?]" if low_confidence else "")
        if fill_percent_volume is not None and abs(fill_percent_volume - fill_percent) > 1.0:
            label_text += f"  (vol {fill_percent_volume:.0f}%)"
        label_y = min((p[1] for p in curve_points), default=float(y))
        _draw_label(
            out, label_text, (x, int(label_y) - 14),
            scale=0.7, fg=(255, 255, 255) if not low_confidence else (20, 20, 20),
            bg=(30, 30, 30) if not low_confidence else (0, 200, 255),
        )

    _draw_label(out, algorithm_label, (10, 24), scale=0.7, fg=(20, 20, 20), bg=(255, 255, 255))
    return out
