"""Generira objedinjene rezultate s jednim retkom po ground-truth boci."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from algoritmi import common, lab_euklidska, rgb_euklidska, kmeans_euklidska, chanvese_euklidska
from evaluacija import metrike
from evaluacija.evaluacija_prozirnost import classify

ROOT = Path(__file__).resolve().parent.parent
# Ground truth defines the evaluated set: 36 images and 65 bottles.
INPUT_DIR = ROOT / "boce"
RESULTS_DIR = ROOT / "evaluacija" / "rezultati"
IOU_MATCH = 0.4
ALGORITHMS = {
    "lab": lab_euklidska,
    "rgb": rgb_euklidska,
    "kmeans": kmeans_euklidska,
    "chanvese": chanvese_euklidska,
}


def decimal(value, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def decimal_text(value: str) -> str:
    return value.replace(".", ",") if value else value


def parse_categories(name: str) -> dict[str, str]:
    tokens = name.rsplit(".", 1)[0].split("_")
    return {
        "broj_boca": tokens[-4],
        "osvjetljenje": tokens[-3],
        "kategorija_punjenja": tokens[-2],
        "slozenost": tokens[-1],
    }


def outline_edges(module, image):
    function = getattr(module, "compute_outline_edges", None)
    if function is None:
        return None
    try:
        return function(image)
    except TypeError:
        return function(image, None)


def detect(module, image):
    if hasattr(module, "_boxes_and_binary"):
        boxes, binary = module._boxes_and_binary(image)
        edges = module.compute_outline_edges(image, binary)
    else:
        boxes = module.detect_bottle_regions(image)
        edges = outline_edges(module, image)

    detections = []
    for box in boxes:
        outline = module.native_outline(image, box, edges)
        _curve, fill, confidence = module.estimate_liquid_level(image, box, outline)
        detections.append((box, fill, confidence))
    return detections


def main() -> None:
    ground_truth = json.loads((ROOT / "evaluacija" / "ground_truth.json").read_text(encoding="utf-8"))
    image_names = sorted(ground_truth)
    bottle_count = sum(len(annotations) for annotations in ground_truth.values())
    if len(image_names) != 36 or bottle_count != 65:
        raise RuntimeError(
            f"Ocekivani skup ima 36 slika i 65 boca, a ground_truth sadrzi "
            f"{len(image_names)} slika i {bottle_count} boca."
        )
    missing = [name for name in image_names if not (INPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Nedostaju slike ground truth skupa u folderu boce: " + ", ".join(missing)
        )
    transparency_path = RESULTS_DIR / "prozirnost_po_boci.csv"
    transparency = {}
    with transparency_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            transparency[(row["slika"], int(row["gt_indeks"]))] = row

    rows = []
    for image_name in image_names:
        annotations = ground_truth[image_name]
        image = common.load_image(INPUT_DIR / image_name)
        height, width = image.shape[:2]
        categories = parse_categories(image_name)
        detections_by_algorithm = {
            key: detect(module, image) for key, module in ALGORITHMS.items()
        }
        for gt_index, annotation in enumerate(annotations):
            gt_box = tuple(annotation["bbox"])
            actual_fill = metrike.ground_truth_fill_percent(gt_box, annotation["level_y"])
            row = {
                "slika": image_name,
                "gt_indeks": gt_index,
                "sirina_px": width,
                "visina_px": height,
                **categories,
                "stvarna_razina_%": decimal(actual_fill, 1),
            }
            transparency_row = transparency.get((image_name, gt_index), {})
            row["prozirnost"] = transparency_row.get("klasa", "")
            row["zasicenje"] = decimal_text(transparency_row.get("zasicenje", ""))
            row["bgDist"] = decimal_text(transparency_row.get("bgDist", ""))
            row["uopce_detektirano"] = "ne"

            for key, detections in detections_by_algorithm.items():
                boxes = [detection[0] for detection in detections]
                matches, unmatched_gt, _unmatched_det = metrike.match_boxes(
                    [gt_box], boxes, IOU_MATCH
                )
                prefix = f"{key}_"
                if matches and not unmatched_gt:
                    _gt_index, detection_index = matches[0]
                    detected_box, fill, confidence = detections[detection_index]
                    row[prefix + "detektirano"] = "da"
                    row[prefix + "procijenjena_razina_%"] = decimal(fill, 1)
                    row[prefix + "iou"] = decimal(metrike.iou(gt_box, detected_box), 3)
                    row[prefix + "pogreska_%"] = decimal(fill - actual_fill, 1)
                    row[prefix + "pouzdanost"] = decimal(confidence, 2)
                    row["uopce_detektirano"] = "da"
                else:
                    row[prefix + "detektirano"] = "ne"
                    row[prefix + "procijenjena_razina_%"] = "nije detektirano"
                    row[prefix + "iou"] = "nije detektirano"
                    row[prefix + "pogreska_%"] = "nije detektirano"
                    row[prefix + "pouzdanost"] = "nije detektirano"
            rows.append(row)

    output = RESULTS_DIR / "rezultati_po_boci.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Spremljeno: {output}")
    print(f"Redaka: {len(rows)}")


if __name__ == "__main__":
    main()