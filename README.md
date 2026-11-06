# GOME-LSIF retrieval code (1995–2011)

This repository contains the Python code for **GOME-LSIF**, an orbit-level solar-induced chlorophyll fluorescence (SIF) dataset at 740 nm derived from the Global Ozone Monitoring Experiment (GOME) aboard ERS-2. The dataset covers **1 July 1995 to 30 June 2011** and is described in the accompanying *Scientific Data* manuscript.

The processing uses GOME Band 4 radiances in the **743–758 nm** window, Mali1-based radiometric degradation correction, dynamically selected non-fluorescent training spectra, an SVD-based retrieval, zero-offset correction, and public-product packaging.

## Processing workflow

The method-level sequence is:

```text
PICS evaluation and degradation-model fitting
  -> scan-position-specific radiometric correction
  -> non-fluorescent training-set construction
  -> SVD-based SIF retrieval
  -> zero-offset fitting and correction
  -> public-product packaging
```

In the implementation, AVHRR-based target classification is performed before the selected GOME training radiances are corrected. Both training spectra and observation spectra receive the degradation correction before retrieval.

| Script | Purpose |
| --- | --- |
| `00_fit_pics_degradation_model.py` | Fit Mali1 reflectance using polynomial trends and six-harmonic seasonal components. |
| `01_select_nonfluorescent_training_samples.py` | Select bare-land, snow/bright-surface and thick-sea-cloud training spectra. |
| `02_correct_training_spectra_for_degradation.py` | Apply the fixed Mali1 quadratic degradation correction to training spectra. |
| `03a_retrieve_sif_1995_2003.py` | Retrieve orbit-level SIF for July 1995–June 2003. |
| `03b_retrieve_sif_2003_2011.py` | Retrieve orbit-level SIF for July 2003–June 2011. |
| `04_fit_zero_offset_model.py` | Fit task- and scan-position-specific zero-offset coefficients. |
| `05_apply_zero_offset_correction.py` | Apply the coefficients to separate corrected orbit files. |
| `06_generate_public_product.py` | Filter, package and verify the GOME-LSIF v1.0 public product. |

`gome_sif_utils.py` provides the file-discovery and gridding functions used by training selection. `retrieve_gome_sif_orbit_Mali1_quadratic_S2_SZA75.py` contains the shared retrieval implementation used by `03a`, `03b`, `04` and `05`. Keep both modules alongside the numbered scripts.

## Inputs

The workflow requires external input data, which are not distributed in this repository:

- ERS-2 GOME Level 1b NetCDF files containing Band 4 radiance, calibrated wavelength, geolocation, solar/viewing geometry and cloud information.
- Daily AVHRR LTDR NetCDF files containing surface reflectance bands `SREFL_CH1` and `SREFL_CH2`, and 11-µm brightness temperature `BT_CH4`, in the scaling expected by the selection script.
- The Mali1 candidate-site file `Mali1_GOME_BAND4_199506_201107_1.5deg.nc`, containing the `CALIBRATION`, `DAILY_IRRADIANCE` and `BAND4_FOOTPRINTS` groups, for degradation fitting.

Run the scripts from the repository root. Default paths are relative to that directory:

```text
data/
  GOME/YYYY/MM/DD/<GOME Level 1b files>
  AVHRR/YYYY/<daily AVHRR NetCDF files>
  PICS/Mali1_GOME_BAND4_199506_201107_1.5deg.nc
outputs/
  degradation/
  training_raw/
  training_corrected/
  pre_zero/
  zero_offset/
  GOME_LSIF_public_v1_0/
```

Path and execution settings are defined near the top of the relevant scripts. Configure these before running and keep stage outputs in separate directories.

## Software requirements

Python **3.10 or later** is required. Direct dependencies are:

```text
numpy
pandas
scipy
netCDF4
xarray
tqdm
scikit-image
scikit-learn
global-land-mask
matplotlib
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

[requirements-verified.txt](requirements-verified.txt) records the installed versions used for verification; it is not a complete environment lock.

## Running the workflow

After configuring the inputs, run:

```bash
python 00_fit_pics_degradation_model.py
python 01_select_nonfluorescent_training_samples.py
python 02_correct_training_spectra_for_degradation.py
python 03a_retrieve_sif_1995_2003.py
python 03b_retrieve_sif_2003_2011.py
python 04_fit_zero_offset_model.py
python 05_apply_zero_offset_correction.py
python 06_generate_public_product.py
```

Stage `00` writes fitted parameters and diagnostics. Stages `02` and `03` use the fixed production quadratic coefficients embedded in their source; they do not automatically load a new fit from `00`. The applied degradation factor is `C(t) = P(0)/P(t)`, referenced to 1 July 1995. Only the long-term quadratic component is applied to spectra; the seasonal Fourier component is excluded from the correction factor.

Retrieval groups use nominally seven available observation dates before July 2003 and 15 thereafter, with tail-merging and later-period training-date borrowing. The full available training-date inventory determines the task schedule. For a preview after preparing that inventory:

```bash
python 03a_retrieve_sif_1995_2003.py --dry-run
python 03b_retrieve_sif_2003_2011.py --dry-run
```

Complete both retrieval periods before fitting the full-period zero-offset model. Check the stage `04` coefficient manifest for `status=PASS` and an empty `blocked_tasks` list before running `05`. A complete successful packaging run requires `run_summary.json` from stage `06` to have `status=PASS`.

## Retrieval settings and training thresholds

The retrieval uses **72 consecutive calibrated spectral samples**, **four SVD components** and **three Legendre terms**. Training and observation geometry use **SZA <75°** and **VZA <60°**. NADIR observation mode includes east, center/nadir and west forward-scan positions.

For each retrieval period and scan position, the implementation requires:

- At least **100 valid candidate training spectra** before balanced sampling.
- At least **100 spectra after sampling**.
- At least **72 spectra after PCA reconstruction screening**.

Retrieval periods failing these requirements are skipped. PCA uses up to 20 components and retains spectra with maximum channel-wise relative reconstruction error <0.005. Bare-land sampling is capped at 160 spectra per 10° latitude band. The initial cloud-to-bare-land target ratio is 1:1; later-period sampling supplements cloud spectra towards **200 training spectra in total**, where available. **200 is a supplementation target, not a universal rejection threshold.**

Zero-offset correction uses a seven-parameter quadratic model in `cos(SZA)`, mean radiance and `cos(latitude)`, without interaction terms:

```text
mu = cos(SZA)
v  = cos(latitude)
sif_bias = mu * (A + B*mu + C*mean_radiance + D*v
                 + E*mu**2 + F*mean_radiance**2 + G*v**2)
sif_740_zero_offset_corrected = sif_740 - sif_bias
```

Angles are converted from degrees to radians before cosine evaluation. Coefficients are fitted separately for each task and scan position using the retained non-fluorescent reference spectra.

## Public-product filtering and use

The public product retains NADIR land observations satisfying:

| Variable or condition | Retained values |
| --- | --- |
| Latitude | >−60° |
| Solar zenith angle | <75° |
| Viewing zenith angle | <60° |
| Reduced chi-square | ≤1.2 |
| `sif_740`, before zero-offset correction | [−5,5], inclusive |
| Degradation-corrected mean radiance over 743–758 nm | [25,200], inclusive |

SIF and radiance units are **mW m⁻² sr⁻¹ nm⁻¹**. The source intermediate retrieval uses `sif_740` in [−5,10]; stage `06` applies the narrower public interval above.

No additional cloud-fraction screening or range filter on the final corrected SIF is applied during packaging. Missing or invalid source cloud values may remain. Empty orbit files, record order and retained source indices are preserved.

The recommended scientific variable is **`sif_740_zero_offset_corrected`**. `sif_740` contains SIF before zero-offset correction, and `sif_bias` is the estimated subtracted bias. `sif_standard_error` describes theoretical retrieval precision and excludes systematic errors and correction-model uncertainty.

Public files are organized as `YYYY/YYYYMMDD/original_GOME_L1b_filename_SIF.nc`. See the [user guide](GOME_LSIF_v1_0_user_guide_1995_2011.txt) for variable definitions and usage notes.

## Examples, references and verification

Run the supplied real-orbit reading example and synthetic checks without the external raw-data archive:

```bash
python examples/read_product.py
python tests/verify_shared_code.py
```

The [reference directory](reference/README.md) contains the checked production task schedule, training counts, zero-offset coefficients and fixed Mali1 parameters. These tables support comparison with a new run; they are not automatically substituted for its generated task, basis or provenance files.

Selected real-data checks reproduced two training tasks and one nonempty orbit per task, with matching production values. The full mission was not rerun. See [VALIDATION.md](VALIDATION.md) for verification results and limits, and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for source records.
