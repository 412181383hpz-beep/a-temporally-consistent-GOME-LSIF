"""Fit GOME FOV-dependent degradation with a joint seasonal/trend model.

Method basis:
    R_hat_s(t) = P_s^p(t) * (1 + F_s^6(t)), p in {1, 2}

Only the polynomial P_s^p(t) is used to construct the degradation and
correction factors.  When enabled, potential outliers are identified from an
initial all-data fit using the residual MAD criterion, retained in the audit
table, and excluded from one final refit.  No iterative clipping is performed.

The script estimates factors near 758 nm and assumes that each FOV-dependent
factor is wavelength-independent over 743-758 nm.  It writes factors only; it
does not modify the source NetCDF or any full radiance spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import xarray as xr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SITES = ["Mali1"]
BASE_PATH = Path("data/PICS")
OUTPUT_BASE_DIR = Path(
    "outputs/degradation"
)
REFERENCE_DATE = pd.Timestamp("1995-07-01")
TARGET_WAVELENGTH_NM = 758.0
FOURIER_ORDER = 6
CLOUD_FRACTION_MAX = 0.3
MIN_FIT_SAMPLES = 30
PLOT_DPI = 800
REMOVE_MAD_OUTLIERS = True

PIXEL_TYPES = {
    0: "East Pixel",
    1: "Center Pixel",
    2: "West Pixel",
}
PIXEL_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}
MODEL_ORDERS = {"linear": 1, "quadratic": 2}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


@dataclass
class FitResult:
    model_name: str
    trend_order: int
    params: np.ndarray
    predicted: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    optimizer_cost: float
    optimizer_nfev: int
    optimizer_message: str


def flatten_to_1d(values: np.ndarray) -> np.ndarray:
    """Average non-footprint dimensions while preserving the first dimension."""
    values = np.asarray(values)
    if values.ndim <= 1:
        return values
    axes = tuple(range(1, values.ndim))
    return np.nanmean(values, axis=axes)


def time_in_years(dates: pd.Series | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    """Return years since the common reference date, using 365.25 d yr-1."""
    date_values = pd.to_datetime(dates)
    return np.asarray((date_values - REFERENCE_DATE) / pd.Timedelta(days=365.25))


def verify_source_pixel_mapping(source_pixel: xr.DataArray) -> None:
    """Validate codes and check whether NetCDF metadata confirms FOV semantics."""
    values = np.asarray(source_pixel.values)
    finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values
    observed = set(np.unique(finite).astype(int).tolist())
    expected = set(PIXEL_TYPES)
    if observed != expected:
        raise ValueError(
            f"source_pixel codes are {sorted(observed)}, expected {sorted(expected)}. "
            "FOV identity cannot be inferred safely."
        )

    attrs = dict(source_pixel.attrs)
    meanings = str(attrs.get("flag_meanings", "")).lower()
    flag_values = attrs.get("flag_values")
    if meanings and flag_values is not None:
        words = meanings.split()
        values_meta = np.asarray(flag_values).astype(int).tolist()
        expected_words = ["east", "center", "west"]
        if len(words) != len(values_meta) or any(
            not any(expected in word for word in words) for expected in expected_words
        ):
            raise ValueError(
                "source_pixel metadata exists but does not unambiguously describe "
                "East/Center/West; refusing to guess the mapping."
            )
        mapping = dict(zip(values_meta, words))
        if any(expected_words[code] not in mapping.get(code, "") for code in sorted(expected)):
            raise ValueError(
                f"source_pixel metadata conflicts with PIXEL_TYPES: {mapping}."
            )
        print("source_pixel metadata confirms the configured East/Center/West mapping.")
    else:
        warnings.warn(
            "source_pixel contains only codes 0, 1, and 2, but its NetCDF attributes "
            "do not define their FOV meanings. The script uses the explicitly configured "
            "0=East, 1=Center, 2=West mapping; this semantic mapping still requires "
            "confirmation from the data producer.",
            RuntimeWarning,
        )


def build_nadir_mask(ds_fp: xr.Dataset) -> tuple[np.ndarray, str | None]:
    """Select nadir mode only when an unambiguous observation-mode variable exists."""
    candidates = ("observation_mode", "measurement_mode", "scan_mode", "mode")
    name = next((candidate for candidate in candidates if candidate in ds_fp.variables), None)
    n = ds_fp.sizes["footprint"]
    if name is None:
        print("No observation-mode variable was found; no nadir-mode filter was invented.")
        return np.ones(n, dtype=bool), None

    variable = ds_fp[name]
    values = flatten_to_1d(variable.values)
    if values.dtype.kind in {"U", "S", "O"}:
        normalized = np.char.lower(values.astype(str))
        mask = np.char.find(normalized, "nadir") >= 0
    else:
        flag_values = variable.attrs.get("flag_values")
        flag_meanings = str(variable.attrs.get("flag_meanings", "")).lower().split()
        nadir_indices = [index for index, meaning in enumerate(flag_meanings) if "nadir" in meaning]
        if flag_values is None or len(nadir_indices) != 1:
            raise ValueError(
                f"Mode variable '{name}' exists, but its numeric metadata does not "
                "identify a nadir code; refusing to guess."
            )
        values_meta = np.asarray(flag_values)
        if len(values_meta) != len(flag_meanings):
            raise ValueError(f"Mode metadata length mismatch for '{name}'.")
        nadir_value = values_meta[nadir_indices[0]]
        mask = values == nadir_value

    if not np.any(mask):
        raise ValueError(f"Mode variable '{name}' was found, but no nadir observations remain.")
    print(f"Applied nadir-mode filter using variable '{name}'.")
    return np.asarray(mask, dtype=bool), name


def calculate_reflectance(
    radiance: np.ndarray,
    irradiance: np.ndarray,
    solar_zenith_angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate pi*L/(E0*cos(SZA)) and return intermediate quantities."""
    cos_sza = np.cos(np.deg2rad(solar_zenith_angle))
    input_valid = (
        np.isfinite(radiance)
        & (radiance > 0)
        & np.isfinite(irradiance)
        & (irradiance > 0)
        & np.isfinite(cos_sza)
        & (cos_sza > 0)
    )
    reflectance = np.full(radiance.shape, np.nan, dtype=float)
    reflectance[input_valid] = (
        np.pi * radiance[input_valid] / (irradiance[input_valid] * cos_sza[input_valid])
    )
    return reflectance, cos_sza, input_valid


def build_quality_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Build transparent, cumulative quality masks in the requested order."""
    result = df.copy()
    result["mask_geometry_valid"] = (
        result["radiance_valid"]
        & result["irradiance_valid"]
        & result["cos_sza_valid"]
        & np.isfinite(result["sza"])
        & np.isfinite(result["vza"])
    )
    result["mask_reflectance_valid"] = (
        result["mask_geometry_valid"]
        & np.isfinite(result["reflectance"])
        & (result["reflectance"] > 0)
    )
    result["mask_after_cloud"] = (
        result["mask_reflectance_valid"]
        & np.isfinite(result["cloud_fraction"])
        & (result["cloud_fraction"] < CLOUD_FRACTION_MAX)
    )
    result["mask_final"] = result["mask_after_cloud"] & result["mask_nadir"]
    return result


def read_site_footprints(netcdf_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the three 758 nm channels and construct footprint-level audit fields."""
    with xr.open_dataset(netcdf_path, group="CALIBRATION", engine="netcdf4") as ds_cal:
        wavelengths = np.asarray(ds_cal["wavelength"].values, dtype=float)
    idx_758 = int(np.nanargmin(np.abs(wavelengths - TARGET_WAVELENGTH_NM)))
    if idx_758 < 1 or idx_758 + 1 >= wavelengths.size:
        raise ValueError("The nearest 758 nm channel does not have two adjacent channels.")
    channel_slice = slice(idx_758 - 1, idx_758 + 2)
    selected_wavelengths = wavelengths[channel_slice]

    with xr.open_dataset(
        netcdf_path, group="DAILY_IRRADIANCE", engine="netcdf4"
    ) as ds_irr:
        irradiance_dates = pd.to_datetime(ds_irr["time"].values).normalize()
        daily_irradiance = np.nanmean(
            np.asarray(ds_irr["IRRADIANCE"].values[:, channel_slice], dtype=float), axis=1
        )
        irradiance_by_date = pd.Series(daily_irradiance, index=irradiance_dates)
        if irradiance_by_date.index.has_duplicates:
            irradiance_by_date = irradiance_by_date.groupby(level=0).mean()

    with xr.open_dataset(
        netcdf_path, group="BAND4_FOOTPRINTS", engine="netcdf4"
    ) as ds_fp:
        required = {
            "radiance",
            "cloud_fraction",
            "solar_zenith_angle",
            "viewing_zenith_angle",
            "source_pixel",
            "time",
        }
        missing = required.difference(ds_fp.variables)
        if missing:
            raise KeyError(f"Missing required BAND4_FOOTPRINTS variables: {sorted(missing)}")

        verify_source_pixel_mapping(ds_fp["source_pixel"])
        nadir_mask, mode_variable = build_nadir_mask(ds_fp)
        radiance = np.nanmean(
            np.asarray(ds_fp["radiance"].values[:, channel_slice], dtype=float), axis=1
        )
        cloud_fraction = flatten_to_1d(ds_fp["cloud_fraction"].values).astype(float)
        sza = flatten_to_1d(ds_fp["solar_zenith_angle"].values).astype(float)
        vza = flatten_to_1d(ds_fp["viewing_zenith_angle"].values).astype(float)
        pixel_type = flatten_to_1d(ds_fp["source_pixel"].values).astype(int)
        times = pd.to_datetime(ds_fp["time"].values)

    dates = times.normalize()
    irradiance = irradiance_by_date.reindex(dates).to_numpy(dtype=float)
    reflectance, cos_sza, input_valid = calculate_reflectance(radiance, irradiance, sza)

    df = pd.DataFrame(
        {
            "datetime": times,
            "date": dates,
            "pixel_type": pixel_type,
            "radiance_758_mean": radiance,
            "irradiance_758_mean": irradiance,
            "reflectance": reflectance,
            "cloud_fraction": cloud_fraction,
            "sza": sza,
            "vza": vza,
            "mask_nadir": nadir_mask,
            "radiance_valid": np.isfinite(radiance) & (radiance > 0),
            "irradiance_valid": np.isfinite(irradiance) & (irradiance > 0),
            "cos_sza_valid": np.isfinite(cos_sza) & (cos_sza > 0),
        }
    )
    df = build_quality_masks(df)
    if not np.array_equal(input_valid, df["radiance_valid"] & df["irradiance_valid"] & df["cos_sza_valid"]):
        raise AssertionError("Reflectance input mask is inconsistent with quality-mask fields.")

    metadata = {
        "selected_wavelengths_nm": selected_wavelengths,
        "mode_variable": mode_variable,
        "data_start": df["date"].min(),
        "data_end": df["date"].max(),
    }
    print(
        "Selected channels near 758 nm: "
        + ", ".join(f"{value:.4f}" for value in selected_wavelengths)
    )
    return df, metadata


def aggregate_daily_by_fov(footprints: pd.DataFrame) -> pd.DataFrame:
    """Create daily FOV statistics while retaining every quality-stage count."""
    keys = ["date", "pixel_type"]
    grouped_all = footprints.groupby(keys, observed=True, sort=True)
    counts = grouped_all.agg(
        n_raw=("reflectance", "size"),
        n_after_geometry_filter=("mask_geometry_valid", "sum"),
        n_reflectance_valid=("mask_reflectance_valid", "sum"),
        n_after_cloud_filter=("mask_after_cloud", "sum"),
        n_after_nadir_filter=("mask_final", "sum"),
        n_valid_footprints=("mask_final", "sum"),
    ).reset_index()

    valid = footprints.loc[footprints["mask_final"]].copy()
    daily_stats = (
        valid.groupby(keys, observed=True, sort=True)
        .agg(
            reflectance_daily_mean=("reflectance", "mean"),
            reflectance_daily_median=("reflectance", "median"),
            reflectance_daily_std=("reflectance", "std"),
            sza_daily_mean=("sza", "mean"),
            vza_daily_mean=("vza", "mean"),
        )
        .reset_index()
    )
    daily = counts.merge(daily_stats, on=keys, how="left", validate="one_to_one")
    daily["fov"] = daily["pixel_type"].map(PIXEL_TYPES)
    daily["t_years"] = time_in_years(daily["date"])
    daily = daily.sort_values(keys).reset_index(drop=True)
    return daily


def polynomial_trend(t: np.ndarray, params: np.ndarray, trend_order: int) -> np.ndarray:
    """Evaluate u0 + u1*t (+ u2*t^2)."""
    return np.polynomial.polynomial.polyval(t, np.asarray(params)[: trend_order + 1])


def joint_model(t: np.ndarray, params: np.ndarray, trend_order: int) -> np.ndarray:
    """Evaluate P^p(t) * [1 + F^6(t)] with no Fourier constant term."""
    t = np.asarray(t, dtype=float)
    params = np.asarray(params, dtype=float)
    start = trend_order + 1
    cos_coeff = params[start : start + FOURIER_ORDER]
    sin_coeff = params[start + FOURIER_ORDER : start + 2 * FOURIER_ORDER]
    seasonal = np.zeros_like(t, dtype=float)
    for harmonic in range(1, FOURIER_ORDER + 1):
        phase = 2.0 * np.pi * harmonic * t
        seasonal += (
            cos_coeff[harmonic - 1] * np.cos(phase)
            + sin_coeff[harmonic - 1] * np.sin(phase)
        )
    return polynomial_trend(t, params, trend_order) * (1.0 + seasonal)


def initial_parameters(t: np.ndarray, y: np.ndarray, trend_order: int) -> np.ndarray:
    """Initialize trend first, then estimate Fourier coefficients from y/P - 1."""
    poly = np.polynomial.polynomial.polyfit(t, y, trend_order)
    trend = np.polynomial.polynomial.polyval(t, poly)
    if np.any(~np.isfinite(trend)) or np.any(trend <= 0):
        poly = np.zeros(trend_order + 1, dtype=float)
        poly[0] = float(np.nanmedian(y))
        trend = np.full_like(y, poly[0], dtype=float)

    design_columns = []
    for harmonic in range(1, FOURIER_ORDER + 1):
        design_columns.append(np.cos(2.0 * np.pi * harmonic * t))
    for harmonic in range(1, FOURIER_ORDER + 1):
        design_columns.append(np.sin(2.0 * np.pi * harmonic * t))
    design = np.column_stack(design_columns)
    fourier, *_ = np.linalg.lstsq(design, y / trend - 1.0, rcond=None)
    fourier = np.clip(fourier, -0.5, 0.5)
    return np.concatenate([poly, fourier])


def fit_joint_model(
    t: np.ndarray,
    y: np.ndarray,
    trend_order: int,
    model_name: str,
) -> FitResult:
    """Fit one multiplicative joint model using ordinary nonlinear least squares."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    k = trend_order + 1 + 2 * FOURIER_ORDER
    if len(y) < max(MIN_FIT_SAMPLES, k + 2):
        raise RuntimeError(
            f"{model_name} fit has {len(y)} samples, fewer than the required "
            f"{max(MIN_FIT_SAMPLES, k + 2)}."
        )
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError(f"{model_name} fit received invalid time or reflectance values.")

    p0 = initial_parameters(t, y, trend_order)
    result = least_squares(
        lambda params: joint_model(t, params, trend_order) - y,
        p0,
        loss="linear",
        method="trf",
        x_scale="jac",
        max_nfev=20000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        raise RuntimeError(
            f"{model_name} fit did not converge: status={result.status}, "
            f"message={result.message}"
        )

    predicted = joint_model(t, result.x, trend_order)
    trend = polynomial_trend(t, result.x, trend_order)
    return FitResult(
        model_name=model_name,
        trend_order=trend_order,
        params=result.x,
        predicted=predicted,
        trend=trend,
        residual=y - predicted,
        optimizer_cost=float(result.cost),
        optimizer_nfev=int(result.nfev),
        optimizer_message=str(result.message),
    )


def calculate_model_metrics(
    y: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int,
) -> dict[str, float | int]:
    """Calculate fit statistics with Gaussian least-squares information criteria."""
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(y)
    residual = y - predicted
    sse = float(np.sum(residual**2))
    rmse = float(np.sqrt(sse / n))
    mae = float(np.mean(np.abs(residual)))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - sse / ss_total) if ss_total > 0 else np.nan
    adjusted_r_squared = (
        float(1.0 - (1.0 - r_squared) * (n - 1) / (n - parameter_count - 1))
        if np.isfinite(r_squared) and n > parameter_count + 1
        else np.nan
    )
    pearson_r = (
        float(np.corrcoef(y, predicted)[0, 1])
        if np.std(y) > 0 and np.std(predicted) > 0
        else np.nan
    )
    safe_sse = max(sse, np.finfo(float).tiny)
    aic = float(n * np.log(safe_sse / n) + 2 * parameter_count)
    aicc = (
        float(aic + 2 * parameter_count * (parameter_count + 1) / (n - parameter_count - 1))
        if n > parameter_count + 1
        else np.nan
    )
    bic = float(n * np.log(safe_sse / n) + parameter_count * np.log(n))
    return {
        "N": n,
        "k": parameter_count,
        "SSE": sse,
        "RMSE": rmse,
        "MAE": mae,
        "pearson_r": pearson_r,
        "R_squared": r_squared,
        "adjusted_R_squared": adjusted_r_squared,
        "AIC": aic,
        "AICc": aicc,
        "BIC": bic,
    }


def calculate_mad_flags(
    residual: np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return robust z scores and MAD flags for the supplied residuals."""
    residual = np.asarray(residual, dtype=float)
    median_residual = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median_residual)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= np.finfo(float).eps:
        warnings.warn(
            f"{context}: residual MAD is zero or invalid; robust z scores are NaN "
            "and no MAD flags are set.",
            RuntimeWarning,
        )
        return np.full(residual.shape, np.nan), np.zeros(residual.shape, dtype=bool), mad, np.nan
    centered = residual - median_residual
    robust_z = centered / robust_sigma
    flags = np.abs(centered) > 3.0 * robust_sigma
    return robust_z, flags, mad, robust_sigma


def calculate_degradation_factors(
    t: np.ndarray,
    params: np.ndarray,
    trend_order: int,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate P(t), D=P(t)/P(0), and correction=P(0)/P(t)."""
    trend = polynomial_trend(np.asarray(t, dtype=float), params, trend_order)
    reference_trend = float(polynomial_trend(np.array([0.0]), params, trend_order)[0])
    if not np.isfinite(reference_trend) or reference_trend <= 0:
        raise ValueError(f"{context}: P(0)={reference_trend!r} is physically invalid.")
    if np.any(~np.isfinite(trend)) or np.any(trend <= 0):
        bad_count = int(np.sum((~np.isfinite(trend)) | (trend <= 0)))
        raise ValueError(
            f"{context}: polynomial trend is non-finite or <= 0 at {bad_count} dates; "
            "invalid correction factors were not calculated."
        )
    degradation = trend / reference_trend
    correction = reference_trend / trend
    return trend, degradation, correction


def leave_one_year_out_cv(
    dates: pd.Series,
    t: np.ndarray,
    y: np.ndarray,
    trend_order: int,
    model_name: str,
) -> dict[str, object]:
    """Refit after leaving out each complete observed calendar year."""
    years = pd.to_datetime(dates).dt.year.to_numpy()
    predictions = np.full(len(y), np.nan, dtype=float)
    failed: list[str] = []
    successful = 0
    k = trend_order + 1 + 2 * FOURIER_ORDER
    for year in sorted(np.unique(years)):
        test = years == year
        train = ~test
        if train.sum() < max(MIN_FIT_SAMPLES, k + 2) or test.sum() == 0:
            failed.append(f"{year}:insufficient_samples")
            continue
        try:
            fit = fit_joint_model(t[train], y[train], trend_order, model_name)
            train_trend = polynomial_trend(t[train], fit.params, trend_order)
            test_trend = polynomial_trend(t[test], fit.params, trend_order)
            if np.any(train_trend <= 0) or np.any(test_trend <= 0):
                raise ValueError("non-positive polynomial trend")
            predictions[test] = joint_model(t[test], fit.params, trend_order)
            successful += 1
        except Exception as exc:  # each failed year is recorded explicitly
            failed.append(f"{year}:{type(exc).__name__}")

    valid = np.isfinite(predictions)
    if not np.any(valid):
        return {
            "cv_RMSE": np.nan,
            "cv_MAE": np.nan,
            "cv_successful_years": 0,
            "cv_failed_years": ";".join(failed),
        }
    errors = y[valid] - predictions[valid]
    return {
        "cv_RMSE": float(np.sqrt(np.mean(errors**2))),
        "cv_MAE": float(np.mean(np.abs(errors))),
        "cv_successful_years": successful,
        "cv_failed_years": ";".join(failed),
    }


def fit_all_models(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, FitResult]], pd.DataFrame, pd.DataFrame]:
    """Fit both trend orders, optionally applying one auditable MAD-based refit."""
    result_daily = daily.copy()
    model_results: dict[int, dict[str, FitResult]] = {}
    metrics_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for model_name in MODEL_ORDERS:
        for prefix in ("predicted", "trend", "residual", "robust_z", "Dfactor", "correction_factor"):
            result_daily[f"{prefix}_{model_name}"] = np.nan
        result_daily[f"initial_predicted_{model_name}"] = np.nan
        result_daily[f"initial_residual_{model_name}"] = np.nan
        result_daily[f"robust_z_initial_{model_name}"] = np.nan
        result_daily[f"mad_flag_initial_{model_name}"] = False
        result_daily[f"mad_flag_{model_name}"] = False
        result_daily[f"used_in_final_fit_{model_name}"] = False

    for pixel_code, fov_name in PIXEL_TYPES.items():
        model_results[pixel_code] = {}
        valid_mask = (
            (result_daily["pixel_type"] == pixel_code)
            & np.isfinite(result_daily["reflectance_daily_mean"])
            & (result_daily["n_valid_footprints"] > 0)
        )
        fitted = result_daily.loc[valid_mask].sort_values("date")
        t = fitted["t_years"].to_numpy(dtype=float)
        y = fitted["reflectance_daily_mean"].to_numpy(dtype=float)

        initial_fits: dict[str, FitResult] = {}
        initial_robust_z: dict[str, np.ndarray] = {}
        initial_flags: dict[str, np.ndarray] = {}
        initial_mad_stats: dict[str, tuple[float, float]] = {}
        for model_name, trend_order in MODEL_ORDERS.items():
            context = f"{fov_name} {model_name}"
            try:
                initial_fit = fit_joint_model(t, y, trend_order, model_name)
            except Exception as exc:
                raise RuntimeError(f"Initial fit failed for {context}: {exc}") from exc

            robust_z_initial, mad_flags, initial_mad, initial_robust_sigma = calculate_mad_flags(
                initial_fit.residual, f"{context} initial fit"
            )
            initial_fits[model_name] = initial_fit
            initial_robust_z[model_name] = robust_z_initial
            initial_flags[model_name] = mad_flags
            initial_mad_stats[model_name] = (initial_mad, initial_robust_sigma)

        common_mad_flags = np.logical_or.reduce(list(initial_flags.values()))
        used_in_final_fit = (
            ~common_mad_flags if REMOVE_MAD_OUTLIERS else np.ones(len(y), dtype=bool)
        )

        for model_name, trend_order in MODEL_ORDERS.items():
            context = f"{fov_name} {model_name}"
            initial_fit = initial_fits[model_name]
            robust_z_initial = initial_robust_z[model_name]
            model_initial_flags = initial_flags[model_name]
            initial_mad, initial_robust_sigma = initial_mad_stats[model_name]
            k = trend_order + 1 + 2 * FOURIER_ORDER
            if used_in_final_fit.sum() < max(MIN_FIT_SAMPLES, k + 2):
                raise RuntimeError(
                    f"{context}: only {used_in_final_fit.sum()} samples remain after MAD filtering."
                )

            if REMOVE_MAD_OUTLIERS and np.any(common_mad_flags):
                try:
                    fit_retained = fit_joint_model(
                        t[used_in_final_fit], y[used_in_final_fit], trend_order, model_name
                    )
                except Exception as exc:
                    raise RuntimeError(f"MAD-filtered refit failed for {context}: {exc}") from exc
                final_predicted = joint_model(t, fit_retained.params, trend_order)
                fit = FitResult(
                    model_name=model_name,
                    trend_order=trend_order,
                    params=fit_retained.params,
                    predicted=final_predicted,
                    trend=polynomial_trend(t, fit_retained.params, trend_order),
                    residual=y - final_predicted,
                    optimizer_cost=fit_retained.optimizer_cost,
                    optimizer_nfev=fit_retained.optimizer_nfev,
                    optimizer_message=fit_retained.optimizer_message,
                )
            else:
                fit = initial_fit

            _, _, final_mad, final_robust_sigma = calculate_mad_flags(
                fit.residual[used_in_final_fit], f"{context} final retained fit"
            )
            trend, degradation, correction = calculate_degradation_factors(
                t, fit.params, trend_order, context
            )

            ordered_index = fitted.index
            result_daily.loc[ordered_index, f"initial_predicted_{model_name}"] = initial_fit.predicted
            result_daily.loc[ordered_index, f"initial_residual_{model_name}"] = initial_fit.residual
            result_daily.loc[ordered_index, f"robust_z_initial_{model_name}"] = robust_z_initial
            result_daily.loc[ordered_index, f"mad_flag_initial_{model_name}"] = model_initial_flags
            result_daily.loc[ordered_index, f"predicted_{model_name}"] = fit.predicted
            result_daily.loc[ordered_index, f"trend_{model_name}"] = trend
            result_daily.loc[ordered_index, f"residual_{model_name}"] = fit.residual
            result_daily.loc[ordered_index, f"robust_z_{model_name}"] = robust_z_initial
            result_daily.loc[ordered_index, f"mad_flag_{model_name}"] = common_mad_flags
            result_daily.loc[ordered_index, f"used_in_final_fit_{model_name}"] = used_in_final_fit
            result_daily.loc[ordered_index, f"Dfactor_{model_name}"] = degradation
            result_daily.loc[ordered_index, f"correction_factor_{model_name}"] = correction
            model_results[pixel_code][model_name] = fit

            initial_metrics = calculate_model_metrics(y, initial_fit.predicted, k)
            metrics = calculate_model_metrics(
                y[used_in_final_fit], fit.predicted[used_in_final_fit], k
            )
            cv = leave_one_year_out_cv(
                fitted.loc[used_in_final_fit, "date"],
                t[used_in_final_fit],
                y[used_in_final_fit],
                trend_order,
                model_name,
            )
            metrics_rows.append(
                {
                    "pixel_type": pixel_code,
                    "fov": fov_name,
                    "model": model_name,
                    "trend_order": trend_order,
                    **metrics,
                    **cv,
                    "initial_N": initial_metrics["N"],
                    "initial_RMSE": initial_metrics["RMSE"],
                    "initial_MAE": initial_metrics["MAE"],
                    "initial_R_squared": initial_metrics["R_squared"],
                    "n_removed_mad": int(np.sum(common_mad_flags)) if REMOVE_MAD_OUTLIERS else 0,
                    "n_initial_mad_flag_model": int(np.sum(model_initial_flags)),
                    "outlier_policy": (
                        "one_pass_initial_residual_MAD_removed"
                        if REMOVE_MAD_OUTLIERS
                        else "MAD_flagged_but_retained"
                    ),
                    "residual_median": float(np.median(fit.residual[used_in_final_fit])),
                    "residual_MAD": final_mad,
                    "robust_sigma": final_robust_sigma,
                    "initial_residual_MAD": initial_mad,
                    "initial_robust_sigma": initial_robust_sigma,
                    "n_mad_flag": int(np.sum(common_mad_flags)),
                    "optimizer_nfev": fit.optimizer_nfev,
                    "optimizer_cost": fit.optimizer_cost,
                    "fit_start": fitted["date"].min(),
                    "fit_end": fitted["date"].max(),
                    "reference_date": REFERENCE_DATE,
                }
            )

            parameter_names = [f"u_{m}" for m in range(trend_order + 1)]
            parameter_names += [f"v_{n}" for n in range(1, FOURIER_ORDER + 1)]
            parameter_names += [f"w_{n}" for n in range(1, FOURIER_ORDER + 1)]
            for parameter_index, (name, value) in enumerate(zip(parameter_names, fit.params)):
                if name.startswith("u_"):
                    parameter_type = "polynomial"
                    harmonic = np.nan
                elif name.startswith("v_"):
                    parameter_type = "fourier_cosine"
                    harmonic = int(name.split("_")[1])
                else:
                    parameter_type = "fourier_sine"
                    harmonic = int(name.split("_")[1])
                parameter_rows.append(
                    {
                        "pixel_type": pixel_code,
                        "fov": fov_name,
                        "model": model_name,
                        "trend_order": trend_order,
                        "fourier_order": FOURIER_ORDER,
                        "parameter_index": parameter_index,
                        "parameter": name,
                        "parameter_type": parameter_type,
                        "harmonic": harmonic,
                        "value": float(value),
                        "reference_date": REFERENCE_DATE,
                        "fit_start": fitted["date"].min(),
                        "fit_end": fitted["date"].max(),
                    }
                )

    return (
        result_daily,
        model_results,
        pd.DataFrame(metrics_rows),
        pd.DataFrame(parameter_rows),
    )


def build_daily_correction_factors(
    model_results: dict[int, dict[str, FitResult]],
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Create continuous daily factors from the reference date to data end."""
    calendar = pd.date_range(REFERENCE_DATE, pd.Timestamp(data_end).normalize(), freq="D")
    t = time_in_years(calendar)
    rows: list[pd.DataFrame] = []
    for pixel_code, fov_name in PIXEL_TYPES.items():
        for model_name, trend_order in MODEL_ORDERS.items():
            fit = model_results[pixel_code][model_name]
            trend, degradation, correction = calculate_degradation_factors(
                t, fit.params, trend_order, f"{fov_name} {model_name} continuous calendar"
            )
            rows.append(
                pd.DataFrame(
                    {
                        "date": calendar,
                        "pixel_type": pixel_code,
                        "fov": fov_name,
                        "model": model_name,
                        "trend_order": trend_order,
                        "fourier_order": FOURIER_ORDER,
                        "trend": trend,
                        "Dfactor": degradation,
                        "correction_factor": correction,
                        "reference_date": REFERENCE_DATE,
                        "factor_source_wavelength_nm": TARGET_WAVELENGTH_NM,
                        "assumed_wavelength_range_nm": "743-758",
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def build_sample_count_tables(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build monthly, annual, and climatological-month sample audits."""
    work = daily.copy()
    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.month
    work["valid_day"] = work["n_valid_footprints"] > 0
    count_columns = [
        "n_raw",
        "n_reflectance_valid",
        "n_after_cloud_filter",
        "n_after_geometry_filter",
        "n_valid_footprints",
    ]

    aggregation: dict[str, tuple[str, str]] = {
        column: (column, "sum") for column in count_columns
    }
    aggregation.update(
        {
            "n_valid_days": ("valid_day", "sum"),
            "n_mad_flag_linear": ("mad_flag_linear", "sum"),
            "n_mad_flag_quadratic": ("mad_flag_quadratic", "sum"),
        }
    )

    monthly = (
        work.groupby(["year", "month", "pixel_type", "fov"], observed=True)
        .agg(**aggregation)
        .reset_index()
    )
    annual = (
        work.groupby(["year", "pixel_type", "fov"], observed=True)
        .agg(**aggregation)
        .reset_index()
    )
    climatological = (
        work.groupby(["month", "pixel_type", "fov"], observed=True)
        .agg(
            n_valid_footprints=("n_valid_footprints", "sum"),
            n_valid_days=("valid_day", "sum"),
            n_mad_flag_linear=("mad_flag_linear", "sum"),
            n_mad_flag_quadratic=("mad_flag_quadratic", "sum"),
        )
        .reset_index()
    )
    return monthly, annual, climatological


def _format_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_fit(
    site: str,
    daily: pd.DataFrame,
    model_results: dict[int, dict[str, FitResult]],
    metrics: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> Path:
    trend_order = MODEL_ORDERS[model_name]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, (pixel_code, fov_name) in zip(axes, PIXEL_TYPES.items()):
        data = daily.loc[
            (daily["pixel_type"] == pixel_code)
            & np.isfinite(daily["reflectance_daily_mean"])
        ].sort_values("date")
        color = PIXEL_COLORS[pixel_code]
        fit = model_results[pixel_code][model_name]
        dense_dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
        dense_t = time_in_years(dense_dates)
        predicted = joint_model(dense_t, fit.params, trend_order)
        trend = polynomial_trend(dense_t, fit.params, trend_order)
        metric = metrics.loc[
            (metrics["pixel_type"] == pixel_code) & (metrics["model"] == model_name)
        ].iloc[0]

        used = data[f"used_in_final_fit_{model_name}"].astype(bool)
        ax.scatter(
            data.loc[used, "date"], data.loc[used, "reflectance_daily_mean"],
            s=10, alpha=0.35, color=color, label="Daily means used in final fit",
        )
        ax.plot(
            dense_dates, predicted, color="black", linewidth=1.2,
            label=f"{fov_name}; p={trend_order}, q={FOURIER_ORDER}; "
            f"$R^2$={metric['R_squared']:.3f}; N={int(metric['N'])}",
        )
        ax.plot(dense_dates, trend, color=color, linewidth=2.0, linestyle="--", label="Polynomial trend")
        flagged = data[f"mad_flag_{model_name}"].astype(bool)
        ax.scatter(
            data.loc[flagged, "date"], data.loc[flagged, "reflectance_daily_mean"],
            s=38, facecolors="none", edgecolors="red", linewidths=1.0,
            label="Initial MAD outlier removed before final refit",
        )
        visible_values = np.concatenate(
            [
                data.loc[used, "reflectance_daily_mean"].to_numpy(dtype=float),
                predicted,
                trend,
            ]
        )
        visible_values = visible_values[np.isfinite(visible_values)]
        if visible_values.size:
            lower = float(np.min(visible_values))
            upper = float(np.max(visible_values))
            padding = max(0.08 * (upper - lower), 0.005)
            ax.set_ylim(lower - padding, upper + padding)
        ax.set_ylabel("Reflectance", fontsize=12)
        ax.legend(loc="best", fontsize=9, framealpha=0.85)
        _format_axes(ax)
    axes[-1].set_xlabel("Date", fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    chinese_name = {
        "linear": "一次趋势联合拟合图",
        "quadratic": "二次趋势联合拟合图",
    }[model_name]
    output_path = output_dir / f"{site}_{chinese_name}.png"
    _save_figure(fig, output_path)
    return output_path


def plot_fit_results(
    site: str,
    daily: pd.DataFrame,
    correction_factors: pd.DataFrame,
    monthly: pd.DataFrame,
    model_results: dict[int, dict[str, FitResult]],
    metrics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Create all requested English, high-resolution diagnostics."""
    paths = [
        _plot_joint_fit(site, daily, model_results, metrics, "linear", output_dir),
        _plot_joint_fit(site, daily, model_results, metrics, "quadratic", output_dir),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (pixel_code, fov_name) in zip(axes, PIXEL_TYPES.items()):
        subset = correction_factors[correction_factors["pixel_type"] == pixel_code]
        for model_name, style in (("linear", "-"), ("quadratic", "--")):
            data = subset[subset["model"] == model_name]
            ax.plot(data["date"], data["trend"], linestyle=style, linewidth=1.8, label=model_name.title())
        ax.set_ylabel("Trend", fontsize=12)
        ax.legend(title=fov_name, fontsize=9)
        _format_axes(ax)
    axes[-1].set_xlabel("Date", fontsize=12)
    paths.append(output_dir / f"{site}_一次与二次长期趋势对比图.png")
    _save_figure(fig, paths[-1])

    for value_column, ylabel, filename in (
        ("Dfactor", "Normalized degradation factor", f"{site}_衰退因子对比图.png"),
        ("correction_factor", "Radiance correction factor", f"{site}_校正因子对比图.png"),
    ):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for ax, (pixel_code, fov_name) in zip(axes, PIXEL_TYPES.items()):
            subset = correction_factors[correction_factors["pixel_type"] == pixel_code]
            for model_name, style in (("linear", "-"), ("quadratic", "--")):
                data = subset[subset["model"] == model_name]
                ax.plot(data["date"], data[value_column], linestyle=style, linewidth=1.8, label=model_name.title())
            ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle=":")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(title=fov_name, fontsize=9)
            _format_axes(ax)
        axes[-1].set_xlabel("Date", fontsize=12)
        path = output_dir / filename
        _save_figure(fig, path)
        paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (pixel_code, fov_name) in zip(axes, PIXEL_TYPES.items()):
        data = daily.loc[
            (daily["pixel_type"] == pixel_code)
            & np.isfinite(daily["reflectance_daily_mean"])
        ].sort_values("date")
        for model_name, color in (("linear", "#1f77b4"), ("quadratic", "#ff7f0e")):
            used = data[f"used_in_final_fit_{model_name}"].astype(bool)
            ax.scatter(
                data.loc[used, "date"], data.loc[used, f"residual_{model_name}"],
                s=8, alpha=0.45, color=color, label=model_name.title(),
            )
            flagged = data[f"mad_flag_{model_name}"].astype(bool)
            ax.scatter(
                data.loc[flagged, "date"], data.loc[flagged, f"residual_{model_name}"],
                s=34, facecolors="none", edgecolors="red", linewidths=0.9,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Residual", fontsize=12)
        ax.legend(title=f"{fov_name}; initial MAD outliers removed", fontsize=9)
        _format_axes(ax)
    axes[-1].set_xlabel("Date", fontsize=12)
    paths.append(output_dir / f"{site}_残差与MAD异常值图.png")
    _save_figure(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_dates = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    for pixel_code, fov_name in PIXEL_TYPES.items():
        mask = monthly["pixel_type"] == pixel_code
        ax.plot(
            monthly_dates[mask], monthly.loc[mask, "n_valid_footprints"],
            color=PIXEL_COLORS[pixel_code], linewidth=1.5, marker="o", markersize=2.5,
            label=fov_name,
        )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Valid footprint count", fontsize=12)
    ax.legend(fontsize=10)
    _format_axes(ax)
    paths.append(output_dir / f"{site}_月度有效样本数量图.png")
    _save_figure(fig, paths[-1])
    return paths


def report_model_comparison(metrics: pd.DataFrame) -> None:
    """Print metric-specific winners without making the scientific choice."""
    print("\nModel comparison (both models are retained):")
    for pixel_code, fov_name in PIXEL_TYPES.items():
        subset = metrics[metrics["pixel_type"] == pixel_code].set_index("model")
        aicc_winner = subset["AICc"].idxmin()
        bic_winner = subset["BIC"].idxmin()
        cv_winner = subset["cv_RMSE"].idxmin() if subset["cv_RMSE"].notna().any() else "unavailable"
        print(
            f"  {fov_name}: AICc -> {aicc_winner}; BIC -> {bic_winner}; "
            f"leave-one-year-out RMSE -> {cv_winner}"
        )
        delta_aicc = abs(subset.loc["linear", "AICc"] - subset.loc["quadratic", "AICc"])
        cv_values = subset["cv_RMSE"].dropna()
        cv_close = (
            len(cv_values) == 2
            and abs(cv_values.iloc[0] - cv_values.iloc[1]) <= 0.05 * max(cv_values.min(), np.finfo(float).eps)
        )
        if delta_aicc < 2 and cv_close:
            print(
                "    AICc differs by <2 and CV errors are within 5%; consider the "
                "simpler linear trend, subject to scientific judgment."
            )
        for model_name in MODEL_ORDERS:
            failed = str(subset.loc[model_name, "cv_failed_years"])
            if failed and failed != "nan":
                print(f"    {model_name} CV failed folds: {failed}")


def validate_results(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
    correction_factors: pd.DataFrame,
) -> None:
    """Run requested logical and numerical consistency checks."""
    if len(metrics) != len(PIXEL_TYPES) * len(MODEL_ORDERS):
        raise AssertionError("Not all three FOVs completed both model fits.")

    count_order = [
        "n_raw",
        "n_after_geometry_filter",
        "n_reflectance_valid",
        "n_after_cloud_filter",
        "n_after_nadir_filter",
        "n_valid_footprints",
    ]
    counts = daily[count_order].to_numpy(dtype=int)
    if np.any(np.diff(counts, axis=1) > 0):
        raise AssertionError("At least one quality-stage count increases instead of decreases.")

    additive_columns = [
        "n_raw",
        "n_reflectance_valid",
        "n_after_cloud_filter",
        "n_after_geometry_filter",
        "n_valid_footprints",
        "n_valid_days",
        "n_mad_flag_linear",
        "n_mad_flag_quadratic",
    ]
    monthly_to_annual = monthly.groupby(["year", "pixel_type"], as_index=False)[additive_columns].sum()
    annual_check = annual[["year", "pixel_type", *additive_columns]].sort_values(["year", "pixel_type"]).reset_index(drop=True)
    monthly_check = monthly_to_annual.sort_values(["year", "pixel_type"]).reset_index(drop=True)
    if not monthly_check.equals(annual_check):
        raise AssertionError("Monthly sample-count sums do not match annual statistics.")

    for row in metrics.itertuples(index=False):
        expected_k = int(row.trend_order + 1 + 2 * FOURIER_ORDER)
        if int(row.k) != expected_k:
            raise AssertionError(f"Incorrect parameter count for {row.fov} {row.model}.")
        parameter_count = len(
            parameters[(parameters["pixel_type"] == row.pixel_type) & (parameters["model"] == row.model)]
        )
        if parameter_count != expected_k:
            raise AssertionError(f"Parameter table is incomplete for {row.fov} {row.model}.")
        initial_n = int(
            np.sum(
                (daily["pixel_type"] == row.pixel_type)
                & np.isfinite(daily["reflectance_daily_mean"])
            )
        )
        final_n = int(
            np.sum(
                (daily["pixel_type"] == row.pixel_type)
                & daily[f"used_in_final_fit_{row.model}"].astype(bool)
            )
        )
        removed_mask = (
            (daily["pixel_type"] == row.pixel_type)
            & daily[f"mad_flag_{row.model}"].astype(bool)
        )
        if int(row.initial_N) != initial_n or int(row.N) != final_n:
            raise AssertionError("Initial or final model sample count is inconsistent.")
        if int(row.n_removed_mad) != initial_n - final_n:
            raise AssertionError("Reported MAD removal count is inconsistent.")
        if daily.loc[removed_mask, f"used_in_final_fit_{row.model}"].any():
            raise AssertionError("At least one MAD-removed point remains in the final fit.")
        if not np.isfinite(row.AICc) or not np.isfinite(row.BIC):
            raise AssertionError("AICc or BIC is invalid.")

    reference = correction_factors[correction_factors["date"] == REFERENCE_DATE]
    if len(reference) != len(PIXEL_TYPES) * len(MODEL_ORDERS):
        raise AssertionError("Reference-date factor rows are incomplete.")
    if not np.allclose(reference["Dfactor"], 1.0, rtol=0, atol=1e-12):
        raise AssertionError("Dfactor is not one on the common reference date.")
    if not np.allclose(reference["correction_factor"], 1.0, rtol=0, atol=1e-12):
        raise AssertionError("correction_factor is not one on the common reference date.")
    if not np.allclose(
        correction_factors["Dfactor"] * correction_factors["correction_factor"],
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("Dfactor * correction_factor differs from one.")
    factor_values = correction_factors[["trend", "Dfactor", "correction_factor"]].to_numpy()
    if np.any(~np.isfinite(factor_values)) or np.any(factor_values <= 0):
        raise AssertionError("Continuous trend/factors contain invalid or non-positive values.")
    required_keys = ["date", "pixel_type", "fov", "model"]
    if correction_factors[required_keys].isna().any().any():
        raise AssertionError("Correction-factor lookup keys contain missing values.")

    print("All numerical, sample-audit, and one-pass MAD-removal checks passed.")


def save_outputs(
    site: str,
    output_dir: Path,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    climatological: pd.DataFrame,
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
    correction_factors: pd.DataFrame,
) -> list[Path]:
    """Write all requested tables with stable names and adequate precision."""
    tables = {
        f"{site}_daily_reflectance_and_residuals.csv": daily,
        f"{site}_monthly_sample_counts.csv": monthly,
        f"{site}_annual_sample_counts.csv": annual,
        f"{site}_climatological_month_counts.csv": climatological,
        f"{site}_fit_parameters.csv": parameters,
        f"{site}_model_metrics.csv": metrics,
        f"{site}_daily_correction_factors.csv": correction_factors,
    }
    paths = []
    for filename, table in tables.items():
        path = output_dir / filename
        table.to_csv(path, index=False, float_format="%.15g", date_format="%Y-%m-%d")
        paths.append(path)
    return paths


def process_site(site: str) -> None:
    """Run the complete fit, audit, output, and plotting workflow for one site."""
    print("\n" + "=" * 78)
    print(f"Processing site: {site}")
    netcdf_path = BASE_PATH / f"{site}_GOME_BAND4_199506_201107_1.5deg.nc"
    if not netcdf_path.exists():
        print(f"ERROR: input file not found: {netcdf_path}")
        return

    output_name = f"{site}_MAD异常值剔除" if REMOVE_MAD_OUTLIERS else site
    output_dir = OUTPUT_BASE_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    footprints, metadata = read_site_footprints(netcdf_path)
    daily = aggregate_daily_by_fov(footprints)
    daily, model_results, metrics, parameters = fit_all_models(daily)
    correction_factors = build_daily_correction_factors(
        model_results, pd.Timestamp(metadata["data_end"])
    )
    monthly, annual, climatological = build_sample_count_tables(daily)
    validate_results(daily, monthly, annual, parameters, metrics, correction_factors)
    report_model_comparison(metrics)

    table_paths = save_outputs(
        site,
        output_dir,
        daily,
        monthly,
        annual,
        climatological,
        parameters,
        metrics,
        correction_factors,
    )
    plot_paths = plot_fit_results(
        site,
        daily,
        correction_factors,
        monthly,
        model_results,
        metrics,
        output_dir,
    )
    print(f"Data range: {metadata['data_start']:%Y-%m-%d} to {metadata['data_end']:%Y-%m-%d}")
    print(f"Output directory: {output_dir}")
    print(f"Wrote {len(table_paths)} CSV files and {len(plot_paths)} figures.")
    print(
        "Radiance correction direction: L_corrected(lambda,t,s) = "
        "L_raw(lambda,t,s) * correction_factor = L_raw / Dfactor."
    )


def main() -> None:
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for site in SITES:
        process_site(site)


if __name__ == "__main__":
    main()
