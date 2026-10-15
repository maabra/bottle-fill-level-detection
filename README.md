# Bottle Fill-Level Detection Using Euclidean Distance

Python implementation accompanying Matej Abramović's master's thesis, **“Analiza algoritama detekcije rubova temeljenih na euklidskoj udaljenosti s primjenom u sustavima industrijske vizualne inspekcije”** (Juraj Dobrila University of Pula, 2026).

This project finds bottles in photographs and estimates how full they are. It compares CIELAB gradients, K-means, and Chan–Vese, with an RGB gradient as a baseline. All four methods use Euclidean distance and run without model training.

The thesis is written in Croatian. Its original filename is `Diplomski_rad_Abramovic_finalno.pdf`.

## Algorithms

| Method | Approach | Implementation |
| --- | --- | --- |
| CIELAB gradient | Local color differences between neighboring pixels identify edges in perceptually uniform CIELAB space. | [lab_euklidska.py](algoritmi/lab_euklidska.py) |
| RGB gradient | The gradient approach in RGB/BGR space, used as an ablation to assess the choice of color space. | [rgb_euklidska.py](algoritmi/rgb_euklidska.py) |
| K-means | Groups pixels by Euclidean distance to cluster centers, using CIELAB color, saturation, and spatial features for localization. | [kmeans_euklidska.py](algoritmi/kmeans_euklidska.py) |
| Chan–Vese | Region-based contour evolution using squared Euclidean distances to mean region colors, with morphological smoothing. | [chanvese_euklidska.py](algoritmi/chanvese_euklidska.py) |

The methods share utilities for finding bottle candidates, searching for the liquid surface in vertical segments, and calculating the filled share of the bottle's bounding-box height. These utilities are in [common.py](algoritmi/common.py).

Roberts, Sobel, Prewitt, Canny, and distance transforms are also discussed and illustrated in the thesis as background comparisons; the quantitative bottle evaluation below covers the four methods above.

## Dataset and results

Evaluation uses **36 photographs containing 65 manually annotated bottles**, with varied lighting, simple and cluttered backgrounds, transparent and colored contents, and single or multiple bottles per image. Reference annotations contain bottle bounding boxes and liquid boundary positions.

Predictions are matched to reference bottles at **IoU ≥ 0.4**. Fill-level MAE and RMSE are measured in **percentage points** on successfully matched bottles; they do not include missed bottles. Average IoU also describes matched detections.

| Method | Matched bottles / 65 | Precision | Recall | Mean IoU | MAE (pp) | RMSE (pp) | Time / image (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIELAB gradient | 32 | 0.696 | 0.492 | 0.752 | 16.69 | 22.72 | 0.255 |
| RGB gradient | 29 | 0.617 | 0.446 | 0.767 | 17.02 | 23.71 | 0.249 |
| K-means | **39** | 0.684 | **0.600** | 0.767 | 15.77 | 21.32 | 2.199 |
| Chan–Vese | 26 | 0.684 | 0.400 | 0.738 | **14.53** | **19.29** | 1.040 |

Source: [saved evaluation summary](evaluacija/rezultati/sazetak.csv). Timings are measurements from the reported experiment and depend on hardware and image size.

![Overall comparison of recall, precision, and fill-level error](pisanje/slike_rada/graf_pregled.png)

K-means finds the most bottles but takes the longest. Chan–Vese has the lowest fill-level error among matched detections, while missing more bottles. The gradient methods are fastest; CIELAB improves precision, recall, and MAE over the RGB baseline. The thesis also reports a median ensemble with **14.2 pp MAE across 44 detected bottles**.

Performance drops in cluttered scenes and on difficult transparent bottles. Lighting categories also differ in scene complexity, so their results do not isolate the effect of lighting alone.

## Visual examples

A simple scene with one bottle:

![Four algorithms on a single-bottle, brightly lit scene](pisanje/slike_rada/usporedba_jednostavna.png)

A cluttered scene with several bottles:

![Four algorithms on a cluttered scene with multiple bottles](pisanje/slike_rada/usporedba_kompleksna.png)

More figures are available in [pisanje/slike_rada](pisanje/slike_rada).

## Run

With Python installed, run these commands from the repository root:

```sh
python -m pip install numpy opencv-python
python main.py
```

Place input images in `boce/`. The interactive menu offers the three main methods, an option to run all three, and a separate RGB ablation. Annotated outputs are saved under `boce_obradeno/`, in a subfolder for each method.

To rerun evaluation against [ground-truth annotations](evaluacija/ground_truth.json):

```sh
python -m evaluacija.pokreni_evaluaciju
```

This writes updated CSV reports to `evaluacija/rezultati/` and appends a run to the evaluation history.
