"""Stage 04: exact saved Stage C rows -> Mali1 quadratic S2-SZA75 cosSZA_cosLat quadratic zero-bias coefficients (no interaction terms)."""
from __future__ import annotations

import csv
import json
import shutil
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import netCDF4 as nc
import numpy as np

# 与 03 放在同一个目录；只共用其反演函数，不会执行 03 的 main。
import retrieve_gome_sif_orbit_Mali1_quadratic_S2_SZA75 as workflow

# 本次范围只筛选固定任务，不重新分组或编号。
RUN_START_DATE = "1995-07-01"
RUN_END_DATE = "2011-06-30"
RUN_TASK_IDS = []  # 空列表表示所选时期的全部固定任务。
train_root = Path("outputs/training_corrected")
gome_root = Path("data/GOME")
product_root = Path("outputs/pre_zero")
TASK_SCHEDULE = product_root / "S2_TASK_SCHEDULE_FIXED_19950701_20110630.csv"
BATCH_NAME = "full_19950701_20110630"  # 每个不同批次使用独立名称；05选择同名系数。
REGRESSION_MODEL = "SIF_zero/cos(SZA)=A+B*cos(SZA)+C*mean_radiance+D*cos(latitude)+E*cos(SZA)**2+F*mean_radiance**2+G*cos(latitude)**2"
output_file = product_root / "_zero_offset_fit_cosSZA_cosLat" / BATCH_NAME / "coefficients.csv"
overwrite = False
dry_run = False
RANDOM_SEED = 20260810
MIN_BIAS_SAMPLES = 100
FOV_NAMES = ("East", "Center", "West")
TYPE_NAMES = {0: "Soil", 1: "Cloud", 2: "Snow"}
PCA_RELATIVE_ERROR_MAX = 0.005
WAVELENGTH_CONVENTION = "First valid NADIR Band-4 calibrated 72-channel grid on Basis_Date; 743-758 nm"


def date_text(path):
    return workflow.date_from_day_folder(path).strftime("%Y-%m-%d")


@lru_cache(maxsize=45)
def load_training_file(path_text):
    with np.load(path_text, allow_pickle=False) as data:
        if set(data.files) != set(workflow.NPZ_KEYS):
            raise ValueError(f"训练六变量不完整: {path_text}")
        arrays = {key: np.asarray(data[key]) for key in data.files}
    n = len(arrays["traindata"])
    if arrays["traindata"].shape != (n, 72) or any(arrays[k].shape != (n,) for k in workflow.NPZ_KEYS[1:]):
        raise ValueError(f"训练shape不一致: {path_text}")
    return arrays


def calibrated_wavelength(day: Path, return_source=False):
    errors: list[str] = []
    for source_path in sorted(day.rglob("*.nc")):
        try:
            with nc.Dataset(source_path) as dataset:
                if workflow.active_nadir_group(dataset) is None:
                    continue
                wavelength_all = workflow.float_array(
                    dataset.groups["CALIBRATION"].variables["wavelength"],
                    (0, 3, slice(None)),
                )
                candidates = np.where(
                    (wavelength_all >= workflow.START_WAVELENGTH_NM)
                    & (wavelength_all <= workflow.END_WAVELENGTH_NM)
                )[0]
                if len(candidates) < workflow.SPECTRAL_CHANNELS:
                    continue
                first = int(candidates[0])
                indices = np.arange(first, first + workflow.SPECTRAL_CHANNELS)
                if indices[-1] >= len(wavelength_all):
                    continue
                wavelength = wavelength_all[indices]
                if (
                    np.all(np.isfinite(wavelength))
                    and np.all(np.diff(wavelength) > 0.0)
                    and wavelength[0] >= workflow.START_WAVELENGTH_NM
                    and wavelength[-1] <= workflow.END_WAVELENGTH_NM
                ):
                    return (wavelength, source_path) if return_source else wavelength
        except Exception as exc:
            errors.append(f"{source_path.name}: {type(exc).__name__}: {exc}")
    suffix = f"; first error: {errors[0]}" if errors else ""
    raise FileNotFoundError(f"No valid Basis_Date wavelength grid in {day}{suffix}")


def retrieve_stage_c(
    matrix: np.ndarray, spectra: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of workflow.retrieve_one_spectrum()."""
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, spectra.T, rcond=None)
    if rank < matrix.shape[1]:
        raise np.linalg.LinAlgError("Rank-deficient SIF retrieval matrix")
    rebuilt = (matrix @ coefficients).T
    residual = spectra - rebuilt
    residual_std = np.std(residual, axis=1)
    mean_radiance = np.mean(spectra, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = mean_radiance / residual_std
        reduced_chi_square = np.sum(
            (rebuilt - spectra) ** 2 / (spectra / snr[:, None]) ** 2,
            axis=1,
        ) / (spectra.shape[1] - matrix.shape[1])
    residual_variance = np.sum(residual**2, axis=1) / (
        spectra.shape[1] - matrix.shape[1]
    )
    inverse_normal = np.linalg.inv(matrix.T @ matrix)
    sif_variance = residual_variance * inverse_normal[-1, -1]
    sif = coefficients[-1]

    check_indices = np.unique(np.linspace(0, len(spectra) - 1, min(3, len(spectra)), dtype=int))
    for index in check_indices:
        formal_sif, formal_chi, formal_variance = workflow.retrieve_one_spectrum(
            matrix, spectra[index]
        )
        if not (
            np.isclose(sif[index], formal_sif, rtol=1e-10, atol=1e-10)
            and np.isclose(reduced_chi_square[index], formal_chi, rtol=1e-10, atol=1e-10)
            and np.isclose(sif_variance[index], formal_variance, rtol=1e-10, atol=1e-10)
        ):
            raise AssertionError("Vectorized retrieval differs from retrieve_one_spectrum")
    return sif, reduced_chi_square, sif_variance


def safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def composition_flag(cloud_fraction: float) -> str:
    if cloud_fraction > 0.90:
        return "cloud_gt90"
    if cloud_fraction > 0.80:
        return "cloud_gt80"
    if cloud_fraction > 0.70:
        return "cloud_gt70"
    return "normal"


def nan_statistics(prefix: str, row: dict[str, object]) -> None:
    names = [
        "A_Intercept", "B_CosSZA", "C_Radiance", "D_CosLatitude",
        "E_CosSZA2", "F_Radiance2", "G_CosLatitude2", "R_squared",
        "RMSE_Normalized", "Residual_Mean_Normalized", "Residual_Median_Normalized",
        "Residual_STD_Normalized", "Residual_MAD_Normalized", "SIF_Zero_Mean",
        "SIF_Zero_Median", "SIF_Zero_STD", "SIF_Zero_Min", "SIF_Zero_Max",
        "Corrected_Zero_Mean", "Corrected_Zero_Median", "Corrected_Zero_STD",
        "CosSZA_Min", "CosSZA_Max", "CosSZA_Mean",
        "CosLatitude_Min", "CosLatitude_Max", "CosLatitude_Mean",
        "Corrected_Zero_RMSE", "SZA_Min", "SZA_Max", "SZA_Mean",
        "Radiance_Min", "Radiance_Max", "Radiance_Mean", "Latitude_Min",
        "Latitude_Max", "Latitude_Mean", "Design_Matrix_Rank",
        "Design_Matrix_Condition_Number", "Soil_Corrected_Zero_Mean",
        "Soil_Corrected_Zero_Median", "Cloud_Corrected_Zero_Mean",
        "Cloud_Corrected_Zero_Median", "Snow_Corrected_Zero_Mean",
        "Snow_Corrected_Zero_Median",
    ]
    del prefix
    row.update({name: float("nan") for name in names})


def basic_row(
    task_index: int,
    training_days: list[Path],
    inversion_days: list[Path],
    basis_path: Path,
    fov_index: int,
    wavelength: np.ndarray | None,
    random_seed: int,
) -> dict[str, object]:
    phase = (
        "Phase 2"
        if workflow.date_from_day_folder(training_days[0]) >= workflow.PHASE_BOUNDARY
        else "Phase 1"
    )
    row: dict[str, object] = {
        "Task_Index": task_index,
        "Phase": phase,
        "Basis_Date": date_text(inversion_days[0]),
        "Basis_Path": str(basis_path),
        "Training_Start": date_text(training_days[0]),
        "Training_End": date_text(training_days[-1]),
        "Training_Day_Count": len(training_days),
        "Training_Dates": ";".join(date_text(day) for day in training_days),
        "Inversion_Start": date_text(inversion_days[0]),
        "Inversion_End": date_text(inversion_days[-1]),
        "Inversion_Day_Count": len(inversion_days),
        "Inversion_Dates": ";".join(date_text(day) for day in inversion_days),
        "Column": fov_index,
        "FOV_Name": FOV_NAMES[fov_index],
        "Random_Seed": random_seed,
        "PCA_Relative_Error_Max": PCA_RELATIVE_ERROR_MAX,
        "Reduced_Chi_Square_Max": workflow.CHI_SQUARE_MAX,
        "Minimum_Bias_Samples": MIN_BIAS_SAMPLES,
        "Regression_Model": REGRESSION_MODEL,
        "Wavelength_Convention": WAVELENGTH_CONVENTION,
        "Wavelength_First_nm": float(wavelength[0]) if wavelength is not None else float("nan"),
        "Wavelength_Last_nm": float(wavelength[-1]) if wavelength is not None else float("nan"),
        "StageC_Total": 0,
        "StageC_Soil": 0,
        "StageC_Cloud": 0,
        "StageC_Snow": 0,
        "StageC_Soil_Fraction": float("nan"),
        "StageC_Cloud_Fraction": float("nan"),
        "StageC_Snow_Fraction": float("nan"),
        "Successful_Retrieval_Count": 0,
        "Fit_Sample_Size": 0,
        "Composition_Flag": "not_available",
        "Status": "retrieval_failed",
        "Error": "",
    }
    nan_statistics("", row)
    return row


def fit_one_fov(
    row: dict[str, object],
    stage_c: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    basis_vectors: np.ndarray,
    formal_count: int,
    wavelength: np.ndarray,
) -> None:
    spectra, latitude, types, sza = stage_c
    total = len(spectra)
    counts = {name: int(np.sum(types == code)) for code, name in TYPE_NAMES.items()}
    row.update(
        StageC_Total=total,
        StageC_Soil=counts["Soil"],
        StageC_Cloud=counts["Cloud"],
        StageC_Snow=counts["Snow"],
        StageC_Soil_Fraction=safe_fraction(counts["Soil"], total),
        StageC_Cloud_Fraction=safe_fraction(counts["Cloud"], total),
        StageC_Snow_Fraction=safe_fraction(counts["Snow"], total),
    )
    row["Composition_Flag"] = composition_flag(row["StageC_Cloud_Fraction"])
    if total != formal_count:
        row["Status"] = "stage_c_count_mismatch"
        row["Error"] = f"reconstructed={total}; basis_training_counts={formal_count}"
        return

    matrix = workflow.retrieval_matrix(basis_vectors, wavelength)
    sif, reduced_chi_square, sif_variance = retrieve_stage_c(matrix, spectra)
    mean_radiance = np.mean(spectra, axis=1)
    successful = (
        np.isfinite(sif)
        & np.isfinite(reduced_chi_square)
        & np.isfinite(sif_variance)
        & (sif_variance >= 0.0)
    )
    row["Successful_Retrieval_Count"] = int(np.sum(successful))
    fit_mask = (
        successful
        & np.isfinite(sza)
        & (sza < workflow.SZA_LIMIT)
        & np.isfinite(latitude)
        & np.isfinite(mean_radiance)
        & (reduced_chi_square <= workflow.CHI_SQUARE_MAX)
    )
    row["Fit_Sample_Size"] = int(np.sum(fit_mask))
    row["_fit_indices"] = np.flatnonzero(fit_mask).tolist()
    if row["Successful_Retrieval_Count"] == 0:
        row["Status"] = "retrieval_failed"
        row["Error"] = "No successful Stage C retrievals"
        return
    if row["Fit_Sample_Size"] < MIN_BIAS_SAMPLES:
        row["Status"] = "insufficient_bias_samples"
        return

    sif_fit = sif[fit_mask]
    sza_fit = sza[fit_mask]
    radiance_fit = mean_radiance[fit_mask]
    latitude_fit = latitude[fit_mask]
    types_fit = types[fit_mask]
    cosine = np.cos(np.deg2rad(sza_fit))
    latitude_predictor = np.cos(np.deg2rad(latitude_fit))
    y = sif_fit / cosine
    design = np.column_stack(
        (np.ones(len(sif_fit)), cosine, radiance_fit, latitude_predictor,
         cosine**2, radiance_fit**2, latitude_predictor**2)
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    condition_number = float(np.linalg.cond(design))
    row["Design_Matrix_Rank"] = int(rank)
    row["Design_Matrix_Condition_Number"] = condition_number
    if rank < 7:
        row["Status"] = "rank_deficient"
        return

    predicted = design @ coefficients
    residual = y - predicted
    centered = y - np.mean(y)
    total_sum_squares = float(np.sum(centered**2))
    residual_sum_squares = float(np.sum(residual**2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    corrected = sif_fit - cosine * predicted
    residual_median = float(np.median(residual))
    row.update(
        A_Intercept=float(coefficients[0]),
        B_CosSZA=float(coefficients[1]),
        C_Radiance=float(coefficients[2]),
        D_CosLatitude=float(coefficients[3]),
        E_CosSZA2=float(coefficients[4]),
        F_Radiance2=float(coefficients[5]),
        G_CosLatitude2=float(coefficients[6]),
        R_squared=r_squared,
        RMSE_Normalized=float(np.sqrt(np.mean(residual**2))),
        Residual_Mean_Normalized=float(np.mean(residual)),
        Residual_Median_Normalized=residual_median,
        Residual_STD_Normalized=float(np.std(residual)),
        Residual_MAD_Normalized=float(np.median(np.abs(residual - residual_median))),
        SIF_Zero_Mean=float(np.mean(sif_fit)),
        SIF_Zero_Median=float(np.median(sif_fit)),
        SIF_Zero_STD=float(np.std(sif_fit)),
        SIF_Zero_Min=float(np.min(sif_fit)),
        SIF_Zero_Max=float(np.max(sif_fit)),
        Corrected_Zero_Mean=float(np.mean(corrected)),
        Corrected_Zero_Median=float(np.median(corrected)),
        Corrected_Zero_STD=float(np.std(corrected)),
        Corrected_Zero_RMSE=float(np.sqrt(np.mean(corrected**2))),
        CosSZA_Min=float(np.min(cosine)),
        CosSZA_Max=float(np.max(cosine)),
        CosSZA_Mean=float(np.mean(cosine)),
        CosLatitude_Min=float(np.min(latitude_predictor)),
        CosLatitude_Max=float(np.max(latitude_predictor)),
        CosLatitude_Mean=float(np.mean(latitude_predictor)),
        SZA_Min=float(np.min(sza_fit)),
        SZA_Max=float(np.max(sza_fit)),
        SZA_Mean=float(np.mean(sza_fit)),
        Radiance_Min=float(np.min(radiance_fit)),
        Radiance_Max=float(np.max(radiance_fit)),
        Radiance_Mean=float(np.mean(radiance_fit)),
        Latitude_Min=float(np.min(latitude_fit)),
        Latitude_Max=float(np.max(latitude_fit)),
        Latitude_Mean=float(np.mean(latitude_fit)),
    )
    for code, name in TYPE_NAMES.items():
        type_values = corrected[types_fit == code]
        row[f"{name}_Corrected_Zero_Mean"] = (
            float(np.mean(type_values)) if len(type_values) else float("nan")
        )
        row[f"{name}_Corrected_Zero_Median"] = (
            float(np.median(type_values)) if len(type_values) else float("nan")
        )
    row["Status"] = "ok"



def load_schedule(path):
    rows = workflow.read_csv(path)
    ids, dates = set(), set()
    for row in rows:
        task = int(row["Task_Index"])
        if task <= 0 or task in ids or int(row["S2_Task_Index"]) != task:
            raise ValueError("任务编号重复或无效")
        ids.add(task)
        inversion = row["Inversion_Dates"].split(";")
        training = row["Training_Dates"].split(";")
        if row["Basis_Date"] != inversion[0] or len(set(training)) != len(training):
            raise ValueError("任务basis/训练日期不一致")
        for date in inversion:
            datetime.strptime(date, "%Y-%m-%d")
            if date in dates:
                raise ValueError("一个反演日期不能对应多个任务")
            dates.add(date)
    if not rows:
        raise ValueError("任务表为空")
    return rows


def day_path(date):
    value = datetime.strptime(date, "%Y-%m-%d")
    return gome_root / f"{value:%Y}" / f"{value:%m}" / f"{value:%d}"


def load_basis_and_lineage(task):
    date = datetime.strptime(task["Basis_Date"], "%Y-%m-%d")
    path = product_root / "_basis" / str(date.year) / f"SVD_Basis_{date:%Y%m%d}.npz"
    qa_path, lineage = path.with_suffix(".qa.json"), path.with_suffix(".lineage.csv")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    inputs = [workflow.train_path_for_date(train_root, datetime.strptime(d, "%Y-%m-%d"))
              for d in task["Training_Dates"].split(";")]
    tables = [p.with_name(p.name.replace("_traindata.npz", "_lineage.csv")) for p in inputs]
    expected = dict(workflow.science_identity(RANDOM_SEED),
                    training_dates=task["Training_Dates"].split(";"),
                    inputs={str(p.resolve()): workflow.file_hash(p) for p in inputs + tables})
    if (not workflow.basis_identity_matches(qa["identity"], expected)
            or qa["basis_sha256"] != workflow.file_hash(path)
            or qa["lineage_sha256"] != workflow.file_hash(lineage)):
        raise ValueError(f"03 basis/训练来源已经变化: {path}")
    with np.load(path, allow_pickle=False) as basis:
        vectors, counts = basis["V_all"], basis["training_counts"]
        if (float(basis["sza_limit"]) != 75 or vectors.shape != (3, 72, 72)
                or counts.shape != (3,) or basis["training_dates"].tolist() != expected["training_dates"]
                or not np.isfinite(vectors).all() or np.any(counts < 72)):
            raise ValueError(f"basis schema/SZA/日期错误: {path}")
    rows = workflow.read_csv(lineage)
    allowed = {str(p.resolve()) for p in inputs}
    for stage in ("StageB", "StageC"):
        identities = set()
        for row in (r for r in rows if r["stage"] == stage):
            source = str(Path(row["source_npz"]).resolve())
            index = int(row["daily_row_index"])
            identity = (source, index)
            if source not in allowed or identity in identities:
                raise ValueError("Stage B/C 来源错误或重复")
            identities.add(identity)
            data = load_training_file(source)
            if not 0 <= index < len(data["traindata"]):
                raise ValueError("Stage C row越界")
            for key, field in (("traincols", "Column"), ("traintype", "traintype"),
                               ("trainlats", "latitude"), ("trainlons", "longitude"), ("trainsza", "SZA")):
                if float(data[key][index]) != float(row[field]):
                    raise ValueError("Stage B/C metadata与当前训练不一致")
    if any(r["stage"] not in ("StageB", "StageC") for r in rows):
        raise ValueError("未知样本stage")
    return path, vectors, counts, rows, [path, qa_path, lineage, *inputs, *tables]


def selected_stage_c(rows, fov):
    selected = [r for r in rows if r["stage"] == "StageC" and int(r["Column"]) == fov]
    if [int(r["stage_row"]) for r in selected] != list(range(len(selected))):
        raise ValueError("Stage C行序与03不一致")
    spectra, lat, types, sza = [], [], [], []
    for row in selected:
        data = load_training_file(str(Path(row["source_npz"]).resolve()))
        i = int(row["daily_row_index"])
        spectra.append(data["traindata"][i])
        lat.append(data["trainlats"][i])
        types.append(data["traintype"][i])
        sza.append(data["trainsza"][i])
    return (np.asarray(spectra, dtype=np.float64), np.asarray(lat, dtype=np.float64),
            np.asarray(types, dtype=np.int8), np.asarray(sza, dtype=np.float64)), selected


def task_artifact_paths(task):
    date = datetime.strptime(task["Basis_Date"], "%Y-%m-%d")
    return [output_file.parent / "lineage" / str(date.year) /
            f'Task_{task["Task_Index"]}_{date:%Y%m%d}_{suffix}.csv'
            for suffix in ("lineage", "sza")]


def checkpoint_path(task):
    return output_file.parent / "checkpoints" / f'Task_{task["Task_Index"]}.json'


def save_checkpoint(task, rows, sources, artifacts, settings):
    workflow.write_json(checkpoint_path(task), dict(
        task=task, rows=rows, sources=sources, artifacts=artifacts, settings=settings))


def read_checkpoint(task, settings, code_sources):
    path = checkpoint_path(task)
    if overwrite or not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["task"] != task or saved["settings"] != settings:
        raise ValueError(f"检查点任务/配置不一致，停止续跑: {path}")
    if any(saved["sources"].get(k) != v for k, v in code_sources.items()):
        raise ValueError(f"检查点源码或固定表已经变化，停止续跑: {path}")
    rows = saved["rows"]
    if (len(rows) != 3 or {int(r["Column"]) for r in rows} != {0, 1, 2}
            or any(int(r["Task_Index"]) != int(task["Task_Index"]) for r in rows)):
        raise ValueError(f"检查点不是完整的task x 3 FOV: {path}")
    # A previously missing basis may now be ready; retry rather than freeze the failure.
    if any(r["Status"] in ("missing_basis", "rebuild_required") for r in rows):
        return None
    for section in ("sources", "artifacts"):
        for name, sha in saved[section].items():
            if not Path(name).exists() or workflow.file_hash(Path(name)) != sha:
                raise ValueError(f"检查点来源/输出已变化，停止续跑: {name}")
    return rows, saved["sources"], saved["artifacts"]


def validate_legacy_task(task, rows):
    """Adopt complete legacy rows after provenance checks, without repeating retrieval."""
    if (len(rows) != 3 or {int(r["Column"]) for r in rows} != {0, 1, 2}
            or any(r["Status"] != "ok" for r in rows)):
        raise ValueError(f'旧部分系数不是完整成功任务: {task["Task_Index"]}')
    if task.get("Rebuild_Required", "0") != "0":
        raise ValueError("旧部分系数对应待重建任务，不能续用")
    path, _, counts, lineage, inputs = load_basis_and_lineage(task)
    training = [day_path(d) for d in task["Training_Dates"].split(";")]
    inversion = [day_path(d) for d in task["Inversion_Dates"].split(";")]
    wavelength, wavelength_source = calibrated_wavelength(inversion[0], return_source=True)
    artifacts = task_artifact_paths(task)
    fitted_lineage = workflow.read_csv(artifacts[0])
    distribution = workflow.read_csv(artifacts[1])
    if any(r["stage"] not in ("StageB", "StageC", "FinalFit")
           or int(r["Task_Index"]) != int(task["Task_Index"])
           or r["Basis_Date"] != task["Basis_Date"]
           or int(r["Column"]) not in (0, 1, 2) for r in fitted_lineage):
        raise ValueError("旧拟合lineage任务/stage错误")
    basis_sha = workflow.file_hash(path)
    for row in rows:
        fov = int(row["Column"])
        expected = basic_row(int(task["Task_Index"]), training, inversion, path,
                             fov, wavelength, RANDOM_SEED)
        # Initial metadata fields precede StageC_Total; diagnostics are validated below.
        for key, value in expected.items():
            if key == "StageC_Total":
                break
            equal = float(row[key]) == value if isinstance(value, (int, float)) else row[key] == str(value)
            if not equal:
                raise ValueError(f"旧系数配置不一致: task={task['Task_Index']}, {key}")
        if (row["Basis_SHA256"] != basis_sha or int(row["S2_Task_Index"]) != int(task["Task_Index"])
                or Path(row["Wavelength_Source"]).resolve() != wavelength_source.resolve()
                or int(row["StageC_Total"]) != int(counts[fov])
                or int(row["Fit_Sample_Size"]) < MIN_BIAS_SAMPLES
                or int(row["Design_Matrix_Rank"]) != 7
                or not all(np.isfinite(float(row[k])) for k in (
                    "A_Intercept", "B_CosSZA", "C_Radiance", "D_CosLatitude",
                    "E_CosSZA2", "F_Radiance2", "G_CosLatitude2"))):
            raise ValueError("旧系数basis/样本数/系数无效")
        original = [r for r in lineage if int(r["Column"]) == fov]
        stored = [r for r in fitted_lineage if int(r["Column"]) == fov and r["stage"] != "FinalFit"]
        expected_lineage = [dict(r, Task_Index=str(task["Task_Index"]), Basis_Date=task["Basis_Date"])
                            for r in original]
        if stored != expected_lineage:
            raise ValueError("旧拟合Stage B/C来源与03不一致")
        stage_c = {(r["source_npz"], r["daily_row_index"]): r
                   for r in expected_lineage if r["stage"] == "StageC"}
        final = [r for r in fitted_lineage if int(r["Column"]) == fov and r["stage"] == "FinalFit"]
        seen = set()
        for order, r in enumerate(final):
            key = (r["source_npz"], r["daily_row_index"])
            expected_row = dict(stage_c.get(key, {}), stage="FinalFit", stage_row=str(order))
            if key in seen or r != expected_row or not float(r["SZA"]) < workflow.SZA_LIMIT:
                raise ValueError("旧FinalFit行不属于原Stage C或重复")
            seen.add(key)
        if len(final) != int(row["Fit_Sample_Size"]):
            raise ValueError("旧FinalFit数量与系数表不一致")
        for stage, n in (("StageC", int(counts[fov])), ("FinalFit", len(final))):
            matches = [r for r in distribution if int(r["Column"]) == fov and r["stage"] == stage
                       and int(r["Task_Index"]) == int(task["Task_Index"])]
            if len(matches) != 1 or int(matches[0]["N"]) != n:
                raise ValueError("旧SZA诊断数量不一致")
    return ({str(p.resolve()): workflow.file_hash(p) for p in inputs + [wavelength_source]},
            {str(p.resolve()): workflow.file_hash(p) for p in artifacts})


def main():
    schedule_path = TASK_SCHEDULE
    tasks = workflow.select_batch(load_schedule(schedule_path), SimpleNamespace(
        run_start=RUN_START_DATE, run_end=RUN_END_DATE, task_ids=RUN_TASK_IDS))
    if not tasks:
        raise ValueError("本次范围没有固定任务")
    if not BATCH_NAME or Path(BATCH_NAME).name != BATCH_NAME:
        raise ValueError("BATCH_NAME必须是单个目录名")
    manifest_path = output_file.with_suffix(".manifest.json")
    sources = {str(p.resolve()): workflow.file_hash(p) for p in (schedule_path, Path(__file__), Path(workflow.__file__))}
    settings = dict(selected_task_ids=[int(t["Task_Index"]) for t in tasks],
                    train_root=str(train_root.resolve()), gome_root=str(gome_root.resolve()),
                    product_root=str(product_root.resolve()), random_seed=RANDOM_SEED, minimum_bias_samples=MIN_BIAS_SAMPLES,
                    regression_model=REGRESSION_MODEL)
    if dry_run:
        print(f"[DRY RUN] {len(tasks)} tasks x 3 FOV -> {output_file}")
        return
    if output_file.exists() and not overwrite:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["settings"] != settings or saved["coefficient_sha256"] != workflow.file_hash(output_file):
            raise ValueError("已有系数配置或内容不一致，请检查后决定是否 overwrite")
        for section in ("sources", "artifacts"):
            for name, sha in saved[section].items():
                if workflow.file_hash(Path(name)) != sha:
                    raise ValueError(f"已有系数的来源/输出已变化: {name}")
        if any(saved["sources"].get(k) != v for k, v in sources.items()):
            raise ValueError("已有系数对应的源码或任务表已变化")
        print(f"[VERIFIED EXISTING] {output_file}")
        return
    code_sources = dict(sources)
    legacy_path = output_file.with_suffix(".legacy.partial.csv")
    partial_path = output_file.with_suffix(".partial.csv")
    legacy_rows = {}
    if not overwrite:
        if not legacy_path.exists() and partial_path.exists():
            shutil.copy2(partial_path, legacy_path)
        if legacy_path.exists():
            for row in workflow.read_csv(legacy_path):
                legacy_rows.setdefault(int(row["Task_Index"]), []).append(row)
            if set(legacy_rows) - set(settings["selected_task_ids"]):
                raise ValueError("旧部分系数含本批次以外的任务，不能合并")
            print(f"[RESUME] 旧部分系数有{len(legacy_rows)}个任务；逐任务核验后复用。", flush=True)
    rows, artifacts, approved_tasks, blocked_tasks = [], {}, [], []
    try:
        for task in tasks:
            task_id = int(task["Task_Index"])
            resumed = read_checkpoint(task, settings, code_sources)
            if resumed is None and not overwrite and task_id in legacy_rows:
                task_rows = legacy_rows[task_id]
                task_sources, task_artifacts = validate_legacy_task(task, task_rows)
                task_sources.update(code_sources)
                task_sources[str(legacy_path.resolve())] = workflow.file_hash(legacy_path)
                save_checkpoint(task, task_rows, task_sources, task_artifacts, settings)
                resumed = task_rows, task_sources, task_artifacts
            if resumed is not None:
                task_rows, task_sources, task_artifacts = resumed
                rows.extend(task_rows)
                sources.update(task_sources)
                artifacts.update(task_artifacts)
                (approved_tasks if all(r["Status"] == "ok" for r in task_rows) else blocked_tasks).append(task_id)
                print(f'[RESUME VERIFIED] task={task_id}, basis={task["Basis_Date"]}: 跳过已完成拟合', flush=True)
                continue
            task_sources, task_artifacts = dict(code_sources), {}
            training = [day_path(d) for d in task["Training_Dates"].split(";")]
            inversion = [day_path(d) for d in task["Inversion_Dates"].split(";")]
            date = datetime.strptime(task["Basis_Date"], "%Y-%m-%d")
            path = product_root / "_basis" / str(date.year) / f"SVD_Basis_{date:%Y%m%d}.npz"
            if task.get("Rebuild_Required", "0") != "0" or not path.exists():
                counts_path = path.with_suffix(".counts.csv")
                reason = "rebuild_required" if task.get("Rebuild_Required", "0") != "0" else "missing_basis"
                if counts_path.exists():
                    task_sources[str(counts_path.resolve())] = workflow.file_hash(counts_path)
                task_rows = []
                for fov in range(3):
                    row = basic_row(task_id, training, inversion, path, fov, None, RANDOM_SEED)
                    row.update(S2_Task_Index=task_id, Status=reason, Error="03 basis not ready")
                    task_rows.append(row)
                save_checkpoint(task, task_rows, task_sources, task_artifacts, settings)
                rows.extend(task_rows)
                sources.update(task_sources)
                blocked_tasks.append(task_id)
                print(f"[BLOCKED] task={task_id}: {reason}")
                continue
            path, vectors, counts, lineage, inputs = load_basis_and_lineage(task)
            wavelength, wavelength_source = calibrated_wavelength(inversion[0], return_source=True)
            for p in inputs + [wavelength_source]:
                task_sources[str(p.resolve())] = workflow.file_hash(p)
            task_rows, fit_lineage, distribution = [], [], []
            for fov in range(3):
                stage_c, selected = selected_stage_c(lineage, fov)
                if len(selected) != int(counts[fov]):
                    raise ValueError("Stage C数量不匹配")
                row = basic_row(task_id, training, inversion, path, fov, wavelength, RANDOM_SEED)
                row.update(S2_Task_Index=task_id, Basis_SHA256=workflow.file_hash(path),
                           Wavelength_Source=str(wavelength_source))
                fit_one_fov(row, stage_c, vectors[fov], int(counts[fov]), wavelength)
                fit_indices = row.pop("_fit_indices", [])
                for r in (r for r in lineage if int(r["Column"]) == fov):
                    fit_lineage.append(dict(r, Task_Index=task_id, Basis_Date=task["Basis_Date"]))
                for order, i in enumerate(fit_indices):
                    fit_lineage.append(dict(selected[i], stage="FinalFit", stage_row=order,
                                            Task_Index=task_id, Basis_Date=task["Basis_Date"]))
                for name, angles in (("StageC", stage_c[3]), ("FinalFit", stage_c[3][fit_indices])):
                    distribution.append(dict(Task_Index=task_id, Column=fov, stage=name, N=len(angles),
                        SZA_LT70=int(np.sum(angles < 70)), SZA_70_75=int(np.sum((angles >= 70) & (angles < 75))),
                        SZA_min=float(np.min(angles)) if len(angles) else "",
                        SZA_max=float(np.max(angles)) if len(angles) else ""))
                task_rows.append(row)
                print(f'[task {task_id}] {FOV_NAMES[fov]}: {row["Status"]}, Fit N={row["Fit_Sample_Size"]}', flush=True)
            for suffix, values in (("lineage", fit_lineage), ("sza", distribution)):
                target = output_file.parent / "lineage" / str(date.year) / f"Task_{task_id}_{date:%Y%m%d}_{suffix}.csv"
                workflow.write_csv(target, values)
                task_artifacts[str(target.resolve())] = workflow.file_hash(target)
            save_checkpoint(task, task_rows, task_sources, task_artifacts, settings)
            rows.extend(task_rows)
            sources.update(task_sources)
            artifacts.update(task_artifacts)
            if all(r["Status"] == "ok" for r in task_rows):
                approved_tasks.append(task_id)
            else:
                blocked_tasks.append(task_id)
        workflow.write_csv(output_file, rows)
        workflow.write_json(manifest_path, dict(status="PASS" if not blocked_tasks else "INCOMPLETE",
            coefficient_file=str(output_file.resolve()), coefficient_sha256=workflow.file_hash(output_file),
            schedule_file=str(schedule_path.resolve()), sources=sources, artifacts=artifacts,
            settings=settings, approved_tasks=approved_tasks, blocked_tasks=blocked_tasks,
            selected_task_ids=[int(t["Task_Index"]) for t in tasks],
            selected_tasks=tasks, expected_rows=len(tasks) * 3))
    except (Exception, KeyboardInterrupt) as exc:
        workflow.write_csv(output_file.with_suffix(".partial.csv"), rows, fields=None if rows else ["Status", "Error"])
        workflow.write_json(output_file.with_suffix(".error.json"),
                            {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    print(f"系数表: {output_file}")
    print(f"可apply任务={approved_tasks}; 禁止apply任务={blocked_tasks}")


if __name__ == "__main__":
    main()
