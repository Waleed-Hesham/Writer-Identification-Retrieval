# Exercise 3 — VLAD Encoding, GMP & Exemplar-SVM Retrieval

Writer/document retrieval pipeline on the **ICDAR17 Historical-WI** dataset:
build a visual-word codebook from local SIFT descriptors, aggregate each
document into a single VLAD vector, and re-rank with Exemplar-SVMs (E-SVM)
for nearest-neighbor retrieval (Top-1 accuracy / mAP).

## Files

| File | Description |
|---|---|
| `exercise3.py` | Full solution, including the bonus tasks (see below). |
| `skeleton1.py` | Base skeleton implementing the core (non-bonus) pipeline. |
| `parmap.py` | Helper for parallelizing the per-exemplar E-SVM training across processes. |
| `icdar17_labels_train.txt` / `icdar17_labels_test.txt` | Filename → writer/class label mappings for the train/test split. |
| `exercise3.txt` | Raw console logs from the experiment runs referenced below. |
| `result.md` | Short summary of the two headline results. |
| `exercise.pdf`, `exercise-01.pdf`, `projcv.pdf`, `individual.pdf` | Assignment/course PDFs. |

Not tracked in git (see `.gitignore`): `icdar17_local_features.zip` 

## Pipeline

1. **Dictionary** — sample local descriptors from training files and cluster
   them with `MiniBatchKMeans` (`K` clusters, default 100) to get a codebook
   (`mus.pkl.gz`).
2. **VLAD encoding** — for each document, assign its local descriptors to the
   nearest cluster (`cv2.BFMatcher`, hard assignment) and accumulate residuals
   per cluster, either via sum pooling or **generalized max pooling (GMP)**
   (`--gmp`, ridge regression per cluster with regularization `--gamma`).
   Optionally apply power normalization (`--powernorm`) and always L2-normalize
   the final vector.
3. **Evaluate** — rank documents by cosine distance between VLAD vectors and
   report Top-1 accuracy and mAP.
4. **Exemplar-SVM re-ranking** — for each test encoding, train a one-vs-rest
   linear SVM against all training encodings as negatives; the learned weight
   vector becomes the new (re-ranked) embedding for that exemplar. Trained in
   parallel via `parmap`.

### Bonus tasks (`exercise3.py` only)

- **(e) `--use_images`** — skip precomputed `.pkl.gz` features and instead
  load raw images, run OpenCV SIFT with keypoint angle forced to 0, and apply
  Hellinger normalization (L1 normalize + signed sqrt).
- **(f) GMP regularization** — ridge-regression-based generalized max pooling
  (`--gmp --gamma <value>`), reusable from both the descriptor-file and
  image-based encoders.
- **(g) `--multivlad`** — build `--n_codebooks` (default 5) independent
  codebooks, concatenate their VLAD encodings, whiten with PCA down to
  `--pca_dim` (default 1000), then evaluate with and without E-SVM re-ranking.

## Usage

```bash
python exercise3.py \
  --labels_train icdar17_labels_train.txt \
  --labels_test  icdar17_labels_test.txt \
  --in_train icdar17_local_features/train \
  --in_test  icdar17_local_features/test \
  --suffix "_SIFT_patch_pr.pkl.gz" \
  --powernorm
```

Useful flags:

- `--gmp --gamma <g>` — enable generalized max pooling with regularization `g`.
- `--use_images --suffix ".jpg"` — compute SIFT on the fly from raw images
  instead of using precomputed descriptor files (point `--in_train`/`--in_test`
  at the image folders).
- `--multivlad --n_codebooks 5 --pca_dim 1000` — multi-codebook VLAD + PCA
  whitening variant.
- `--k <K>` — number of codebook clusters (default 100).
- `--overwrite` — ignore cached `.pkl.gz` intermediates and recompute.

Intermediate results (`mus.pkl.gz`, `enc_train*.pkl.gz`, `enc_test*.pkl.gz`,
per-codebook/PCA caches for multi-VLAD) are cached to disk and reused on
subsequent runs unless `--overwrite` is passed.

## Results

From `exercise3.txt` (ICDAR17 Historical-WI, 1182 train / 3600 test docs):

| Config | Top-1 | mAP |
|---|---|---|
| baseline VLAD | 0.8236 | 0.6314 |
| + power norm | 0.8236 | 0.6314 |
| + power norm + E-SVM | 0.8883 | 0.7533 |
| + power norm + GMP (γ=1) + E-SVM | 0.8886 | 0.7536 |
| + power norm + GMP (γ=0.1) + E-SVM | 0.8900 | 0.7561 |
| `--use_images` + power norm + E-SVM | 0.8958 | 0.7663 |
| `--use_images` + power norm + GMP (γ=1) + E-SVM | 0.9042 | 0.7803 |
| `--multivlad` + power norm | 0.8808 | 0.7431 |
| `--multivlad` + power norm + E-SVM | 0.8867 | 0.7536 |
| `--use_images` + `--multivlad` + power norm | 0.8908 | 0.7567 |
| `--use_images` + `--multivlad` + power norm + E-SVM | 0.8969 | 0.7673 |

E-SVM re-ranking gives the largest single boost; on-the-fly SIFT extraction
(`--use_images`) with GMP + E-SVM gives the best overall result.

## Requirements

`numpy`, `scikit-learn`, `opencv-python` (with SIFT support), `tqdm`,
`progressbar` (used by `parmap.py`).
