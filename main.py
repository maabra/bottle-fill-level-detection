"""Glavna skripta: detekcija razine tekućine u bocama pomoću tri
algoritma temeljena na euklidskoj udaljenosti, po jedan iz svake
paradigme njenog korištenja:

1. LOKALNA (diferencijalna)  - gradijent boje u CIELAB prostoru:
   euklidska udaljenost SUSJEDNIH piksela (rub = velika udaljenost);
2. GLOBALNA (particijska)    - K-means segmentacija: euklidska
   udaljenost piksela do K CENTROIDA (regija = bliski pikseli);
3. VARIJACIJSKA (energetska) - Chan-Vese aktivna kontura: euklidska
   udaljenost piksela do STATISTIKE REGIJE (kontura minimizira sumu
   kvadratnih odstupanja od prosjeka svake regije).

Uz to je dostupna ablacijska varijanta algoritma 1 u RGB prostoru -
usporedba prostora boja unutar iste (gradijentne) metode.

Ulazne slike (jpg, jpeg, png, bmp, webp, tif/tiff) čitaju se iz foldera
'boce', a obrađene slike s označenom bocom i algoritamski određenom
razinom tekućine spremaju se u zaseban podfolder unutar 'boce_obradeno'
za svaki algoritam.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from algoritmi import common, rgb_euklidska, lab_euklidska, kmeans_euklidska, chanvese_euklidska

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "boce"
OUTPUT_DIR = BASE_DIR / "boce_obradeno"

ALGORITHMS = {
    "1": lab_euklidska,
    "2": kmeans_euklidska,
    "3": chanvese_euklidska,
}


def run_algorithm(module, image_paths):
    out_dir = OUTPUT_DIR / module.OUTPUT_DIRNAME
    common.ensure_dir(out_dir)

    print(f"\n--- {module.ALGORITHM_NAME} ---")
    for path in image_paths:
        try:
            out_path, results = module.process_image(path, out_dir)
        except ValueError as exc:
            print(f"{path.name}: {exc}")
            continue

        if not results:
            print(f"{path.name}: nije pronađena nijedna boca")
            continue

        print(f"{path.name} -> {out_path.name} ({len(results)} boca/e)")
        for result in results:
            line = f"    razina tekućine: {result.fill_percent:.1f}%"
            if result.fill_percent_volume is not None and abs(result.fill_percent_volume - result.fill_percent) > 1.0:
                line += f"  (volumen: {result.fill_percent_volume:.1f}%)"
            print(line)


def print_menu():
    print("\nOdaberite algoritam za detekciju razine tekućine:")
    print("  1 - Gradijent boje u CIELAB prostoru (lokalna paradigma)")
    print("  2 - K-means segmentacija, CIELAB + S (globalna paradigma)")
    print("  3 - Chan-Vese aktivna kontura (varijacijska paradigma)")
    print("  4 - Pokreni sva tri algoritma")
    print("  5 - Ablacija: gradijent boje u RGB prostoru")
    print("  0 - Izlaz")


def main():
    image_paths = common.list_images(INPUT_DIR)
    if not image_paths:
        print(f"Nema podržanih slika u folderu: {INPUT_DIR}")
        return

    print(f"Pronađeno {len(image_paths)} slika u '{INPUT_DIR.name}'.")

    while True:
        print_menu()
        choice = input("Izbor: ").strip()

        if choice == "0":
            break
        elif choice in ALGORITHMS:
            run_algorithm(ALGORITHMS[choice], image_paths)
        elif choice == "4":
            for module in ALGORITHMS.values():
                run_algorithm(module, image_paths)
        elif choice == "5":
            run_algorithm(rgb_euklidska, image_paths)
        else:
            print("Nepoznat izbor, pokušajte ponovno.")


if __name__ == "__main__":
    main()
