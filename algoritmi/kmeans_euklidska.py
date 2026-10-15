"""Algoritam 3: detekcija ruba boce i razine tekućine pomoću K-means
segmentacije temeljene na (kvadriranoj) euklidskoj udaljenosti piksela do
centroida klastera u CIELAB + zasićenje (S) prostoru.

I lokalizacija boce i granica tekućine grade se K-means klasteriranjem
(dakle euklidskom udaljenošću do centroida), samo na različitim
razinama:
- lokalizacija: cijela slika klasterira se u K_LOCALIZATION klastera, a
  maska svakog klastera prosljeđuje se u `common.regions_from_binary_edges`
  koja iz nje (spajanjem fragmentiranih dijelova i geometrijskim
  filtriranjem) izvlači kandidate oblika boce;
- razina tekućine: pikseli unutar okvira boce dijele se u K_LOCAL=2
  klastera ("zrak/prazan prostor" i "tekućina"), a granica se traži kao
  redak koji najbolje dijeli udio "tekućinom ispunjenog" klastera po
  stupcima, potvrđen stvarnim vertikalnim gradijentom (boca je 3D objekt
  pa granica nije nužno ravna crta - vidi `common.boundary_curve`)."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import common

ALGORITHM_NAME = "Euklidska udaljenost - K-means (CIELAB + S)"
OUTPUT_DIRNAME = "kmeans_euklidska"

K_BOUNDARY = 8  # klastera za kartu rubova (obris): više klastera = više klaster-granica
                # = potpuniji obris za praćenje. Kalibrirano (6-16) NAKON čišćenja obrisa
                # na SAMO klaster-granice (bez gradijenta zasićenja): K=6 daje presiromašan
                # obris (MAE 17.9), K=8 najbolji (10.9), 10-16 opet lošije (previše sitnih
                # granica od teksture). Čista particijska vrijednost; usporedba: nekadašnja
                # fuzija s gradijentom zasićenja davala je 9.0 ali je gradijent tuđa paradigma.
K_LOCALIZATION = 5  # testirano 3-7 punom evaluacijom: s 3 klastera boca se zna stopiti s
                    # pozadinskim klasterom (velika mrlja -> loši okviri, IoU ~0.5, TP 13),
                    # s 5 se odvoji (TP 15, MAE 16.2%->11.9%); 6-7 opet lošije (20%) jer se
                    # tijelo boce razlomi na previše klastera pa okviri gube cjelinu
K_LOCAL = 3  # klastera za razinu tekućine. Bilo 2 (zrak/tekućina), ali kod boca s TRI
             # vizualne zone (npr. etiketa ili tinta stakla uz zrak i tekućinu) 2-means
             # je etiketu/staklo stopio s tekućinom pa je rez odlutao. S 3 klastera se
             # "tekućinski" (donji) klaster čisto odvoji od etikete/stakla. Kalibrirano
             # (2-4) na GT okvirima nakon uklanjanja gradijentnog sidrenja: čista razina
             # MAE 12.07(k=2)->10.13(k=3)->14.21(k=4); k=3 gotovo dostiže staru sidrenu
             # točnost (9.99) uz PUNU particijsku čistoću.
SPATIAL_WEIGHT = 0.5  # težina prostornih (x,y) koordinata u LOKALIZACIJSKOM klasteriranju
    # (vidi common.lab_saturation_spatial_features): boce iste boje na različitim mjestima
    # inače padnu u isti klaster i spoje se u jednu (pre)široku regiju - dodavanje pozicije
    # ih razdvaja. Kalibrirano punom evaluacijom (0.3-0.8): 0.5 najbolji (preciznost 0.79->0.91,
    # IoU 0.72->0.77, MAE 13.3->12.5); 0.8 previše (prostor nadjača boju, MAE 17). Prostorne
    # značajke SAMO u lokalizaciji - procjena razine (K_LOCAL) i obris (K_BOUNDARY) ostaju
    # čisti CIELAB+S (tamo pozicija nema smisla: razina je JEDNA horizontala, obris je boja).
MIN_CLUSTER_BOX_AREA_RATIO = 0.12  # stroži prag SAMO za lokalizacijske kandidate iz klastera
    # (zajednički common.MIN_BOTTLE_AREA_RATIO=0.008 je namjerno permisivan za rub-signale koji
    # rijetko daju sitne lažne kandidate) - klasteriranje cijele slike zna razlomiti pozadinu i
    # bocu na više sitnih mrlja koje pojedinačno prođu opći prag. Prije prostornih značajki bio
    # je 0.18 (viši prag nužan jer su se boce stapale u velike mrlje pa su pozadinski fragmenti
    # bili glavni sitni kandidati); OTKAD prostorne značajke (SPATIAL_WEIGHT) razdvajaju boce i
    # drže preciznost visokom (0.91), niži prag propušta i manje STVARNE detekcije bez povratka
    # lažnih - snižen na 0.12 (MAE 12.5->11.8, IoU 0.765->0.770, FP nepromijenjen); 0.10 malo
    # niži MAE ali bliže pozadinskim fragmentima (0.009-0.073 izmjereno), 0.12 sigurniji.
# Prozirne/čiste boce (staklo/plastika) kojima se u boju klasterira SAMO obojana
# tekućina pri dnu (čep/grlo/prazno tijelo su bezbojni pa upadnu u pozadinski
# klaster) daju bocoliku regiju POVRŠINE 0.045-0.07 - iznad općeg praga, ispod
# 0.12 - pa ih je jednolični prag odbacivao (dijagnosticirano: 181058 IoU 0.51,
# 181157 srednja boca IoU 0.51 - stvarne boce, bačene). Takva regija se ipak
# razlikuje od pozadinske trake (zid/kut, koja isto zna biti bocolikog oblika)
# UNUTARNJOM STRUKTUROM: stvarna boca (i prozirna) ima teksturu (odsjaji, rub
# tekućine, navoj, etiketa) i zasićenje-rub, pozadinska traka je glatka.
# Zato se regija površine [CLUSTER_STRUCT_MIN_AREA_RATIO, MIN_CLUSTER_BOX_AREA_RATIO)
# zadržava SAMO ako ima dovoljno TEKSTURE (var_cov). Veće regije (>=0.12) prolaze
# bezuvjetno - nula promjene na već ispravnim detekcijama.
# VAŽNO (puni set 65 boca): raniji uvjet je bio "tekstura I zasićenje-rub", ali
# je zasićenje-rub odbacivao TAMNO STAKLO (npr. tamnoplava boca 222545: var_cov
# 0.17 ali sat_cov 0.005 - tamno staklo ima jaku teksturu, ali gotovo nikakvo
# zasićenje, kao i glatka pozadina). Izmjereno na SVIM boca u složenim scenama:
# var_cov >= 0.124 UVIJEK (min 0.124, median 0.334), a glatka pozadinska FP-traka
# var_cov 0.045 - VAR SAM čisto razdvaja. Sat uvjet uklonjen (odbacivao stvarne
# tamne boce bez dobitka u preciznosti). Prag je NEOSJETLJIV u 0.10-0.15 (sve
# stvarne boce iznad, sva odbačena pozadina ispod) - 0.12 s marginom. Recall
# 0.585->0.600 (vraćena tamnoplava boca), preciznost ~ista, jednostavne scene
# netaknute.
CLUSTER_STRUCT_MIN_AREA_RATIO = 0.045  # donji prag za strukturirane (bocolike) male regije
CLUSTER_STRUCT_MIN_VAR = 0.12   # min. udio teksturiranih piksela (stvarne boce >=0.124, FP-traka 0.045)
SEARCH_RANGE = (0.15, 0.90)  # testirano suzavanje do (0.30,0.75) - prosjek MAE izgleda bolje ali je varljivo:
                             # umjetno povlaci NEODREDJENE (bez signala) procjene prema centru USKOG
                             # raspona, sto slucajno pomogne najgorim slucajevima ali PSUJE vec dobre
                             # (provjereno: 180813 isao s ~1pb greske na +27pb) - odbaceno, original je ispravniji
MARGIN_RATIO = 0.18  # testirano 0.10-0.25, 0.18 ostaje najbolji
N_SEGMENTS = 12  # kalibrirano grid-searchom (9-21); prosjek MAE sva 4 algoritma pao s 10.19% (9) na 9.77%


def edge_map(image_bgr: np.ndarray) -> np.ndarray:
    """Karta rubova izvedena iz K-means segmentacije (za vizualizaciju,
    poglavlje 3): rub je svaka granica između dva različita klastera,
    umjesto klasičnog gradijenta."""
    labels, _centers = common.kmeans_cluster(common.lab_saturation_features(image_bgr), K_BOUNDARY)
    edge = np.zeros(labels.shape, dtype=np.uint8)
    edge[:, :-1][labels[:, :-1] != labels[:, 1:]] = 255
    edge[:-1, :][labels[:-1, :] != labels[1:, :]] = 255
    return edge


TEXTURE_FALLBACK_K = 2               # dvoklasterska particija TEKSTURE (glatka pozadina / teksturirani
                                     # foreground); k=3 "uzmi najjači" bio prekrut (uhvati samo najjaču
                                     # pod-regiju, npr. čep/etiketu - 181058 IoU pao na 0.51), k=2 uzme
                                     # CIJELO teksturirano tijelo boce (181058/175708 IoU 0.97)
TEXTURE_FALLBACK_MIN_AREA_RATIO = 0.045  # stvarne boce >=0.108 povrsine; prag ispod, ali iznad sitnog suma


def texture_cluster_candidate_boxes(image_bgr, existing):
    """Fallback za PROZIRNE boce (staklo/plastika) koje se bojno stope s
    pozadinom pa ih primarni color-K-means UOPĆE ne vidi (nijedan klaster ne
    formira bocoliku regiju - dijagnosticirano: 175708 voda 0 regija, 181157
    lijeva/desna boca 0 regija). Tijelo takve boce je ipak KOHERENTNA regija
    VISOKE TEKSTURE (odsjaji, nabori plastike, navoj čepa, refrakcija svjetla)
    naspram glatke studijske pozadine. Ista K-means PARTICIJA, samo nad
    značajkom TEKSTURE (lokalna varijanca) umjesto boje - partition paradigma,
    drugi feature; ne posuđuje gradijent/Chan-Vese. Klaster(i) s natprosječnom
    teksturom centroida = foreground, iz njega bocolike regije; zadržavaju se
    samo dovoljno velike koje color-prolaz nije već pokrio. Izmjereno: 175708
    IoU 0.97, sve 3 boce u 181157 (0.89/0.97/0.95), 181058 0.97."""
    texture = common._raw_variance_map(image_bgr).astype(np.float64)
    labels, centers = common.kmeans_cluster(texture[:, :, None], TEXTURE_FALLBACK_K)
    high = np.flatnonzero(centers[:, 0] > float(np.median(centers[:, 0])))
    mask = common.exclude_skin((np.isin(labels, high).astype(np.uint8) * 255), image_bgr)
    boxes = common.merge_and_dedup_boxes(common.regions_from_binary_edges(mask))
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    # Zadržava se teksturni kandidat OSIM ako je već gotovo posve SADRŽAN u nekom
    # postojećem okviru (color-prolaz ga je već pokrio). VAŽNO: ako color-prolaz
    # dade samo DJELOMIČAN okvir (npr. donja obojana polovica prozirne boce -
    # 181058 IoU 0.51), teksturni okvir puni boce ga NADILAZI (sadržanost < 0.85)
    # pa se zadržava, a naknadni containment-dedup (common.dedup_boxes) potom
    # odbaci djelomični okvir kao sadržan u punom - tako pobjeđuje potpuniji
    # okvir. Da se koristio samo IoU-preklop (kao prije), puni bi okvir bio
    # odbačen jer se s djelomičnim preklapa (IoU 0.50), i ostao bi lošiji okvir.
    return [b for b in boxes
            if b[2] * b[3] >= TEXTURE_FALLBACK_MIN_AREA_RATIO * img_area
            and all(common._contained_ratio(b, e) < common.CONTAINMENT_DEDUP_RATIO for e in existing)]


def _filter_cluster_boxes(boxes, image_bgr, img_area):
    """Zadržava kandidate iz primarnog K-means prolaza: velike (>=
    MIN_CLUSTER_BOX_AREA_RATIO) bezuvjetno, male (>= CLUSTER_STRUCT_MIN_AREA_RATIO)
    samo ako imaju bocoliku unutarnju TEKSTURU (var_cov) - time se hvataju i
    prozirne i tamno-staklene boce (regija 0.045-0.12), a odbacuju glatke
    pozadinske trake iste veličine. Karta teksture računa se lijeno (samo ako
    postoji mali kandidat)."""
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
    """Lokalizacija ne koristi granice klastera (previše osjetljivo na
    sitnu teksturu pozadine poput fuga pločica), nego same klastere kao
    ispunjene mrlje: svaki klaster (skupina piksela bliskih u euklidskoj
    udaljenosti u CIELAB+S + PROSTOR prostoru) tretira se kao kandidat za
    oblik boce, a `common.regions_from_binary_edges` spaja njegove eventualno
    fragmentirane dijelove (npr. tijelo boce prekinuto natpisom na etiketi
    koji pripada drugom klasteru) prije nego što geometrijski filtri odbace
    one koji ne nalikuju uskom, izduženom tijelu boce.

    Prostor značajki je proširen prostornim koordinatama
    (`common.lab_saturation_spatial_features`, `SPATIAL_WEIGHT`) - bez toga
    se boce ISTE boje na različitim mjestima (red istovjetnih boca) svrstaju
    u JEDAN klaster i spoje u jednu preširoku regiju koju filtri odbace,
    pa se gubi cijeli red odjednom; s pozicijom se razdvajaju."""
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    labels, _centers = common.kmeans_cluster(
        common.lab_saturation_spatial_features(blurred, SPATIAL_WEIGHT), K_LOCALIZATION
    )

    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    boxes: List[common.BBox] = []
    for k in range(K_LOCALIZATION):
        mask = (labels == k).astype(np.uint8) * 255
        # Ruka koja drži bocu za grlo/čep zna se svrstati u isti klaster
        # kao boca (ili u susjedni koji se onda spoji preko
        # merge_stacked_boxes) - uklanjanje boje kože prije traženja
        # kontura sprječava to kod izvora (paradigm-neutralno čišćenje).
        mask = common.exclude_skin(mask, image_bgr)
        boxes.extend(common.regions_from_binary_edges(mask))
    boxes = common.merge_and_dedup_boxes(boxes)
    # Sitni fragmenti (boca razlomljena na previše klastera, ili šum
    # pozadine) odbacuju se PRIJE grubljeg prolaza - velike regije bezuvjetno,
    # male (prozirne boce s obojanom tekućinom) samo ako imaju bocoliku
    # unutarnju strukturu - vidi MIN_CLUSTER_BOX_AREA_RATIO / CLUSTER_STRUCT_*.
    boxes = _filter_cluster_boxes(boxes, image_bgr, img_area)

    # Drugi, GRUBLJI prolaz K-means klasteriranja (k=3) - PARADIGMATSKI
    # ISTOVRSTAN dodatni izvor (K-means, ne tuđa paradigma): providna boca
    # s gradijentom osvjetljenja razdvoji se na K_LOCALIZATION=5 preko više
    # sitnih klastera (svaki premalen da prođe filtre), dok grublje
    # klasteriranje zadrži cijelo tijelo u jednom komadu. Sigurnosni filtri
    # (površina/tekstura/najveći) u color_cluster_candidate_boxes.
    extra_coarse = common.color_cluster_candidate_boxes(image_bgr, boxes, k=3)
    if extra_coarse:
        boxes = sorted(boxes + extra_coarse, key=lambda b: b[0])

    # Fallback za PROZIRNE boce koje ni fini ni grubi color-prolaz ne vide:
    # K-means particija TEKSTURE (partition paradigma, drugi feature). Dodaje
    # samo bocolike, dovoljno velike regije koje color-prolazi nisu pokrili.
    extra_texture = texture_cluster_candidate_boxes(image_bgr, boxes)
    if extra_texture:
        boxes = sorted(boxes + extra_texture, key=lambda b: b[0])

    # Bez posuđenog tijesnjenja okvira (Chan-Vese je varijacijska, tuđa
    # paradigma; vlastito klaster-silueta tijesnjenje izmjereno NETO
    # štetnim - znalo je pokvariti već-dobar klaster-okvir).
    boxes = common.dedup_boxes(list(boxes), max_boxes=len(boxes) or 1)
    # Fizikalno dovršavanje do vrha boce (čep/grlo): klaster tijela zna
    # stati na ramenu (bistri čep pripada drugom klasteru ili pozadini) -
    # `extend_boxes_top` produži okvir prema gore dok ima teksture (odsjaj,
    # navoj, rub čepa). Poznata slijepa točka ostaje bijeli čep na bijeloj
    # pozadini (nema teksture) - vidi common.extend_boxes_top.
    return common.extend_boxes_top(image_bgr, boxes)


def compute_outline_edges(image_bgr: np.ndarray) -> np.ndarray:
    """Karta rubova = ČISTE granice K-means klastera (particijska
    paradigma): rub je svaki prijelaz između dva različita klastera
    (`edge_map`). NE fuzionira se s gradijentom zasićenja (koji je lokalni/
    diferencijalni, tuđ signal) - obris ostaje isključivo particijski.
    Boja kože se uklanja (paradigm-neutralno čišćenje) da se u obris ne
    uvuče ruka."""
    return common.exclude_skin(edge_map(image_bgr), image_bgr)


def native_outline(image_bgr: np.ndarray, bbox: common.BBox, outline_edges: np.ndarray = None) -> "np.ndarray | None":
    """Paradigmatski-vlastit obris: `common.bottle_outline` (paradigm-
    neutralan geometrijski tragač granice) primijenjen na kartu KLASTER-
    granica (`compute_outline_edges`) - izvor karte je K-meansov, pa je
    praćeni obris particijski-native, bez posudbe gradijenta ili Chan-Vese
    siluete. `outline_edges` je predizračunata karta klaster-granica (u
    evaluaciji i process_image se računa jednom po slici). Kad praćenje ne
    uspije, pada na pravokutnik (outline=None)."""
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

    # "Tekućinski" klaster = najčešći u DONJIH 25% visine (BOTTOM_BAND_RATIO).
    # Prije je bilo 5%, ali tanka traka pri samom dnu zna biti etiketa/sjena/
    # odsjaj baze (npr. 180652 etiketa, ili tinta zelenog stakla) pa je
    # bottom_cluster pogađao krivu zonu i rez je odlutao (181009 zeleno staklo:
    # +48 p.b.!). Šira donja traka robustno pogodi stvarnu tekućinu (izmjereno
    # GT okviri: MAE 9.62->8.74, max 41.8->30.9; 181009 +48->+4). Pretpostavka:
    # tekućina ispunjava barem ~pola trake (>~12% visine); u ovom skupu najniža
    # ispunjenost je 19.7% pa vrijedi - kod gotovo prazne boce bi traka bila
    # većinom zrak i identifikacija bi pala (poznato ograničenje, van skupa).
    margin = int(w * MARGIN_RATIO)
    core_labels = labels[:, margin: w - margin] if w - 2 * margin > 4 else labels
    bottom_rows = max(1, int(h * 0.25))
    bottom_cluster = np.bincount(core_labels[-bottom_rows:].ravel()).argmax()

    feature_map = (labels == bottom_cluster).astype(np.float64)[..., None]

    search_range = common.body_search_range(outline, bbox, SEARCH_RANGE)
    # ČISTO PARTICIJSKA razina: rez se traži ISKLJUČIVO na udjelu "tekućinskog"
    # klastera po stupcu (K_LOCAL). Gradijentni pomoćnici (edge_boost, prior
    # meniskusa, label_band) su NAMJERNO isključeni - oni su tuđa (lokalna/
    # gradijentna) paradigma i ulazili bi u samu odluku o retku (množe/dodaju
    # ocjenu, vetiraju retke), pa razina više ne bi bila čisto particijska.
    # Ostaje SAMO paradigm-neutralni fizikalni prior vodoravnosti (linija
    # umjesto parabole + granica odstupanja u common.boundary_curve), koji NE
    # ovisi o gradijentu. (ISPROBANO i odbačeno: maskiranje na vlastiti
    # K-means obris - obris je prečesto nepouzdan pa je maska pogoršala
    # MAE 10.13->10.46.) Cijena čistoće: mjestimično malo nagnute linije.
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
