# a-temporally-consistent-GOME-SIF
# GOME SIF retrieval code (1995-2011)

This repository contains the Python code used to generate the orbit-level Global Ozone Monitoring Experiment (GOME) solar-induced chlorophyll fluorescence (SIF) product described in the accompanying *Scientific Data* manuscript. The processing uses GOME Band 4 radiances in the 743-758 nm window, field-of-view-specific radiometric degradation correction, dynamically selected non-fluorescent spectra, a singular value decomposition (SVD)-based retrieval, zero-offset correction, and public-product packaging.

## Processing workflow

The method-level sequence is: PICS evaluation and degradation-model fitting -> field-of-view-specific radiometric correction -> non-fluorescent training-set construction -> SVD retrieval -> zero-offset fitting and correction. In the supplied implementation, target classification is performed before the selected GOME training radiances are corrected; this does not change the AVHRR-, cloud-, land-, and geometry-based selection rules. The corrected spectra are the inputs to SVD training and retrieval.

`gome_sif_utils.py` contains shared file, gridding, filtering, and least-squares retrieval functions. `legacy_retrieve_sif_without_degradation_correction.py` is retained only as a legacy comparison script and is not part of the production workflow.

## Inputs

The code expects the following external data, which are not distributed in this repository:

- ERS-2 GOME Level 1b files containing Band 4 radiance, calibration wavelength, geolocation, viewing/solar geometry, time, and cloud information.
- Daily AVHRR LTDR surface reflectance bands 1 and 2 and 11-µm brightness temperature for training-sample selection.
- Candidate-site NetCDF files containing the `CALIBRATION`, `DAILY_IRRADIANCE`, and `BAND4_FOOTPRINTS` groups for PICS degradation fitting.

The scripts use the original working-directory layout and contain Windows paths such as `F:\GOME`. Change `GOME_ROOT`, `TRAIN_ROOT`, `SAVE_ROOT`, and related path variables before running. The expected main structure is:

```text
GOME_ROOT/
  YYYY/
    MM/
      DD/
        <GOME Level 1b files>
AVHRR_ROOT/
  YYYY/
    <daily AVHRR NetCDF files>
```

## Software requirements

Python 3.10 or later is recommended. The scripts import the following packages:

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

Example installation:

```bash
python -m pip install numpy pandas scipy netCDF4 xarray tqdm scikit-image scikit-learn global-land-mask matplotlib
```

## Running the workflow

Scripts 00-05 use configuration variables near the top of each file. After setting those paths, run:

```bash
python 00_fit_pics_degradation_model.py
python 01_select_nonfluorescent_training_samples.py
python 02_correct_training_spectra_for_degradation.py
python 03a_retrieve_sif_1995_2003.py
python 03b_retrieve_sif_2003_2011.py
python 04_fit_zero_offset_model.py
python 05_apply_zero_offset_correction.py
```



