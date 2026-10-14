# From wall shear stress to near-wall velocity: a 1-D U-Net, held to honest baselines

Wall models in RANS and LES see only the wall. These two notebooks ask how much the wall shear stress alone tells about the turbulent flow above it. The data are 5,777 y–x planes of the streamwise velocity u from a direct numerical simulation of turbulent channel flow (Re_τ ≈ 125; 107 snapshots × up to 60 spanwise planes, 201 × 384 points). Stage 1 computes the wall gradient τ_w(x) at both walls on the stretched grid and trains a 1-D U-Net to map the 384-point τ_w(x) to u(x) at y+ ≈ 5 and 10, with five exercises. Stage 2 asks the two questions a careful engineer would ask next: how far from the wall the wall shear still informs the flow (y+ = 5 up to the channel centre, against ridge regression on the same input and the correct first-order wall law), and whether the result holds on later snapshots and on spanwise planes the model has not seen. Every training is repeated with three seeds.

Portfolio page: https://vamsigudipati.github.io/wall-shear-to-near-wall-velocity.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_wall_to_near_wall_unet.ipynb` | Stage 1 — From wall shear stress to near-wall velocity with a 1-D U-Net | `results/01_wall_to_near_wall_unet/exercises/` | `models/01_wall_to_near_wall_unet/` (the base U-Net, 8.3 MB, and its normalisation statistics) | `animations/01_*` |
| `notebooks/02_height_and_generalisation.ipynb` | Stage 2 — How far from the wall, and does it generalise? | `results/02_height_and_generalisation/` | — (nine U-Nets trained and scored in the notebook, not saved) | — |

Each notebook is also exported as HTML next to it, with all outputs. `exercises/` holds the tables (CSV) and figures (PNG) of Stage 1's exercises and three-seed repeats (`seeds_runs.csv`, `seeds_summary.csv`); Stage 2 writes its tables and figures directly into its results folder. `animations/` holds the training video and its final frame.

The outputs in the notebooks come from my single run of each notebook. For this repository I changed only text afterwards: a header cell and a paths cell in each notebook, the path lines (data, models, outputs), comments, labels and printed text. The notebooks were not executed again in this layout; the paths cell is therefore the one cell without an output.

## Data

No new upload: the planes are the same 5,777 files as in my U-Net project and come with that repository's release, https://github.com/vamsigudipati/unet-turbulent-fields:

| Release file | Size | Goes to | SHA-256 |
| --- | --- | --- | --- |
| `u_yx_planes_h5.tar` (from the unet-turbulent-fields release) | 1.6 GB | extract into `release_assets/`, giving `release_assets/u_yx_planes_h5/` (5,777 files `P125_83pi.<snapshot>.<plane>.h5`) | `410518b833edc2fd5bf1b5c569fd2ed35ea15824ed924117958ef8c1edefadd5` |

`release_assets/` is ignored by git. The trained U-Net of Stage 1 (8.3 MB) is in git; every model was trained by my runs and no pretrained model is used.

## Setting up and running

I ran everything on a laptop (Apple M3 Max, 48 GB) with Python 3.12, TensorFlow 2.18.1 and Keras 3.6. The frozen environment is in `requirements.txt` (it includes `tensorflow-metal`, which these notebooks do not use); the video uses the ffmpeg binary that `imageio-ffmpeg` installs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name py312-tf-metal-studio
tar -xf u_yx_planes_h5.tar -C release_assets/
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 01_wall_to_near_wall_unet.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 02_height_and_generalisation.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there. On the laptop, Stage 1 took 81 minutes and Stage 2 97 minutes, one after the other.

**Why the CPU.** Before training, I timed one training step of each network inside `model.fit` (batch 32, median of 30 steps after 10 warm-up steps) on the CPU and on the Apple GPU (`results/02_height_and_generalisation/device_timing_*.csv`). In milliseconds per step: the 1-D U-Net 15.6 vs 81.2, the 5-output U-Net 15.6 vs 88.8, the 32-filter U-Net 39.6 vs 94.0, a single convolution block 1.3 vs 6.2, a small dense network 0.74 vs 4.5. The CPU is faster for every network, so both notebooks hide the GPU.

**Why three seeds.** A single training gives one draw of the initial weights and of the batch order. Every training here takes under 15 minutes (Stage 1's U-Net 11.5 minutes), so each is repeated with the seeds 0, 1 and 2 (seed 0 is the base run), and I report the mean, the standard deviation and the range. A difference smaller than that spread is not an effect. Stage 1's U-Net turned out very repeatable (0.0111 ± 0.00004 at y+ = 5); in Stage 2 the range of the U-Net on held-out snapshots is what decides whether it beats ridge at a given height.

## What the stages show

- Stage 1: the U-Net predicts u at y+ ≈ 5 and 10 with a mean absolute error of 0.0111 and 0.0362 (stored units, three seeds), far below the notebook's linear-law baseline. That baseline is wrong, though: it leaves out the viscosity and flips sign at the top wall. Written correctly, u ≈ (du/dy)·(y − y_wall), the wall law is about 2 to 3 times worse than the U-Net, not a thousand.
- Stage 2, height: on later snapshots the model never saw, the U-Net's error is 0.19 of the target's spread at y+ = 5, 0.39 at 10 and 0.61 at 15, better than ridge regression on the same wall values (0.22, 0.48, 0.71) by more than its seed range; by y+ = 30 the advantage is within the seed spread (0.95 against 0.98), and at the channel centre neither beats predicting the mean. The wall shear informs the viscous sublayer and the lower buffer layer, not the flow beyond.
- Stage 2, generalisation: the random split of Stage 1 puts both walls and all spanwise planes of a snapshot on both sides and flatters both learned models by 10–17 %; held-out later snapshots and held-out spanwise planes give the same, higher error (0.39 at y+ = 10).
- Stage 2's U-Nets were capped at 150 epochs and all reached the cap, so they may improve with longer training; ridge, the wall law and the mean are deterministic.
