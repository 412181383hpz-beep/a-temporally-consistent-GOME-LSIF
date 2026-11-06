"""S2-SZA75 Mali1 quadratic SVD and orbit retrieval; standalone VSCode script."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from global_land_mask import globe
from scipy.special import eval_legendre
from sklearn.decomposition import PCA

# 总任务时期固定；分批时只修改 RUN 日期或 RUN_TASK_IDS。
TASK_START_DATE = "1995-07-01"
TASK_END_DATE = "2011-06-30"
RUN_START_DATE = "1995-07-01"
RUN_END_DATE = "2011-06-30"

overwrite = False
RUN_TASK_IDS = []       # 空列表按日期选组；例如 [97, 98]，仍受 RUN 日期约束。
REBUILD_TASK_IDS = []   # 共享版从独立空目录运行，不请求历史任务重建。
DEFAULT_TRAIN_ROOT = Path("outputs/training_corrected")
DEFAULT_GOME_ROOT = Path("data/GOME")
DEFAULT_OUTPUT_ROOT = Path("outputs/pre_zero")
overwrite = False
dry_run = False
max_orbits = None       # None 处理全部所选轨道；小规模试运行可设为 1。
random_seed = 20260810

REFERENCE_DATE = datetime(1995, 7, 1)
TROPICAL_YEAR_DAYS = 365.25
FOV_NAMES = ('East', 'Center', 'West')
QUADRATIC_COEFFICIENTS = np.array([
    [0.592357459173782, -0.0043444841216953, 0.0001508133816801],
    [0.603129648659204, -0.0020531580350824, 0.0001603956822971],
    [0.601390869937673, 0.0021333828147409, 0.00000532247230799871],
], dtype=np.float64)
NPZ_KEYS = ('traindata', 'traincols', 'traintype', 'trainlons', 'trainlats', 'trainsza')


def correction_factors(file_date):
    t = (file_date - REFERENCE_DATE).total_seconds() / (86400.0 * TROPICAL_YEAR_DAYS)
    factors = []
    for u0, u1, u2 in QUADRATIC_COEFFICIENTS:
        trend = u0 + u1 * t + u2 * t**2
        if not np.isfinite(trend) or trend <= 0.0:
            raise ValueError(f'{file_date}: 无效的 Mali1 多项式值')
        factors.append(float(u0 / trend))
    return np.array(factors, dtype=np.float64)


MISSION_START = datetime(1995, 6, 28)
MISSION_END = datetime(2011, 7, 2)
PHASE_BOUNDARY = datetime(2003, 7, 1)
START_WAVELENGTH_NM = 743.0
END_WAVELENGTH_NM = 758.0
SPECTRAL_CHANNELS = 72
SVD_COMPONENTS = 4
LEGENDRE_TERMS = 3
FOV_COUNT = 3
SZA_LIMIT = 75.0
VZA_LIMIT = 60.0
LATITUDE_MIN = -60.0
SIF_MIN = -5.0
SIF_MAX = 10.0
CHI_SQUARE_MAX = 1.2
LATITUDE_BIN_DEGREES = 10
SOIL_BIN_CAP = 160
CLOUD_SOIL_RATIO = 1.0
PHASE2_MIN_SAFE_TOTAL = 200
RADIANCE_CONVERSION_CONSTANT = 1.986446e-9
ORBIT_TIME_PATTERN = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-start", default=RUN_START_DATE)
    parser.add_argument("--run-end", default=RUN_END_DATE)
    parser.add_argument("--task-ids", nargs="+", type=int, default=RUN_TASK_IDS)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--gome-root", type=Path, default=DEFAULT_GOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=overwrite)
    parser.add_argument("--dry-run", action="store_true", default=dry_run)
    parser.add_argument("--max-orbits", type=int, default=max_orbits)
    parser.add_argument("--random-seed", type=int, default=random_seed)
    return parser.parse_args()


def active_nadir_group(dataset: nc.Dataset):
    mode = dataset.groups.get("MODE_NADIR")
    if mode is None or "BAND_4" not in mode.groups:
        return None
    return mode


def stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def date_from_day_folder(path: Path) -> datetime:
    return datetime(int(path.parent.parent.name), int(path.parent.name), int(path.name))


def train_path_for_date(train_root: Path, value: datetime) -> Path:
    return train_root / str(value.year) / f"{value:%Y%m%d}_traindata.npz"


def selected_day_folders(
    gome_root: Path, train_root: Path, start: datetime, end: datetime
) -> list[Path]:
    folders: list[Path] = []
    for year in range(start.year, end.year + 1):
        year_root = gome_root / str(year)
        if not year_root.exists():
            continue
        for month_root in sorted(path for path in year_root.iterdir() if path.is_dir()):
            if not month_root.name.isdigit():
                continue
            for day_root in sorted(path for path in month_root.iterdir() if path.is_dir()):
                if not day_root.name.isdigit():
                    continue
                try:
                    value = date_from_day_folder(day_root)
                except ValueError:
                    continue
                if start <= value <= end and train_path_for_date(train_root, value).exists():
                    folders.append(day_root)
    return sorted(folders, key=date_from_day_folder)


def fixed_groups(days: list[Path], size: int, minimum_tail: int) -> list[list[Path]]:
    groups = [days[index : index + size] for index in range(0, len(days), size)]
    if len(groups) > 1 and len(groups[-1]) < minimum_tail:
        groups[-2].extend(groups[-1])
        groups.pop()
    return groups


def production_tasks(days: list[Path]) -> list[tuple[list[Path], list[Path]]]:
    """Return (training days, inversion days) while preserving legacy phase rules."""

    all_days = sorted(days, key=date_from_day_folder)
    by_month: dict[tuple[int, int], list[Path]] = {}
    for path in all_days:
        value = date_from_day_folder(path)
        by_month.setdefault((value.year, value.month), []).append(path)

    tasks: list[tuple[list[Path], list[Path]]] = []
    for key in sorted(by_month):
        month_days = sorted(by_month[key], key=date_from_day_folder)
        first_date = date_from_day_folder(month_days[0])
        if first_date < PHASE_BOUNDARY:
            for group in fixed_groups(month_days, size=7, minimum_tail=4):
                tasks.append((group, group))
            continue

        for inversion_group in fixed_groups(month_days, size=15, minimum_tail=5):
            if len(inversion_group) >= 15:
                training_group = inversion_group
            else:
                # Borrow earlier valid days across month boundaries.  This keeps
                # the terminal 2011-07-01/02 orbits processable instead of losing
                # them solely because July is a two-day partial mission month.
                last_index = all_days.index(inversion_group[-1])
                training_group = all_days[max(0, last_index - 14) : last_index + 1]
            if len(training_group) < 7:
                print(
                    f"[skip group] {date_from_day_folder(inversion_group[0]):%Y-%m-%d}: "
                    f"only {len(training_group)} valid training days"
                )
                continue
            tasks.append((training_group, inversion_group))
    return tasks


def pca_keep_mask(spectra: np.ndarray) -> np.ndarray:
    components = min(20, spectra.shape[0], spectra.shape[1])
    pca = PCA(n_components=components, svd_solver="full")
    transformed = pca.fit_transform(spectra)
    reconstructed = pca.inverse_transform(transformed)
    relative_error = np.abs(spectra - reconstructed) / (np.abs(spectra) + 1e-10)
    return np.max(relative_error, axis=1) < 0.005


def pca_filter(spectra: np.ndarray) -> np.ndarray:
    return spectra[pca_keep_mask(spectra)]


def balanced_training_indices(
    latitudes: np.ndarray,
    types: np.ndarray,
    phase2: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    bins: list[dict[str, np.ndarray]] = []
    for lower in np.arange(-90, 90, LATITUDE_BIN_DEGREES):
        within = (latitudes >= lower) & (latitudes < lower + LATITUDE_BIN_DEGREES)
        bins.append(
            {
                "soil": np.where((types == 0) & within)[0],
                "cloud": np.where((types == 1) & within)[0],
                "snow": np.where((types == 2) & within)[0],
            }
        )

    selected: list[int] = []
    soil_total = 0
    snow_total = 0
    for item in bins:
        soil = item["soil"]
        sampled = rng.choice(soil, SOIL_BIN_CAP, replace=False) if len(soil) > SOIL_BIN_CAP else soil
        selected.extend(sampled.tolist())
        selected.extend(item["snow"].tolist())
        soil_total += len(sampled)
        snow_total += len(item["snow"])

    cloud_target = int(soil_total * CLOUD_SOIL_RATIO)
    if phase2 and soil_total + snow_total + cloud_target < PHASE2_MIN_SAFE_TOTAL:
        available = sum(len(item["cloud"]) for item in bins)
        cloud_target = min(PHASE2_MIN_SAFE_TOTAL - soil_total - snow_total, available)

    remaining = max(cloud_target, 0)
    sorted_bins = sorted(bins, key=lambda item: len(item["cloud"]))
    for index, item in enumerate(sorted_bins):
        bins_left = len(sorted_bins) - index
        quota = remaining // bins_left if bins_left else 0
        take = min(len(item["cloud"]), quota)
        if take:
            selected.extend(rng.choice(item["cloud"], take, replace=False).tolist())
        remaining -= take
    return np.asarray(selected, dtype=np.int64)



class InsufficientTraining(RuntimeError):
    """Expected scientific sample-count gate; do not run dependent retrieval."""


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def execution_source_hash():
    # Canonical source hash excludes only batch selectors, not scientific settings.
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8-sig"))
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
        {"RUN_START_DATE", "RUN_END_DATE", "RUN_TASK_IDS", "REBUILD_TASK_IDS"} for t in n.targets))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def science_identity(random_seed):
    return {"source_sha256": execution_source_hash(), "random_seed": random_seed,
            "sza_limit": SZA_LIMIT, "vza_limit": VZA_LIMIT,
            "soil_bin_cap": SOIL_BIN_CAP, "cloud_soil_ratio": CLOUD_SOIL_RATIO,
            "phase2_target": PHASE2_MIN_SAFE_TOTAL}


def basis_identity_matches(saved, current):
    if saved == current:
        return True
    # Compatibility is limited to known source versions and this reviewed code.
    known_sources = {
        # Verified VSCode history: changes only add controlled rebuild orchestration.
        "a9b0aa57f21ecd42d09899e685d10246616e1d7bb7789711cc488e303d44444b",
        # Source immediately before this compatibility-only update (rebuilt 333/334).
        "733b3f01aa091025cdbbdd8d74aa4c3d368007e1406caeb24e04ec0bb49ab9c8",
        "4ec25110d05aee7fc6fbccb5ee89c46fcc032e1a973c053547de877a8ac50db6",
        "77efb3cfd2131b4f9ad2add4a77da3afa7f2d24c0dda9cea5bcb35fabc005b03",
        "7c59c7c4054cc31ae5189a7358e00e9650cf0da6173d26134be6557e06ad0042",
        "33012c938c0ca00eae269c658c63a54cf96a9ef8a77cc8b07e0b94dffa1aaa3d",
        "39a9b46ab303bdc58f955d243de888eeea988ab17c24824d352ed8707e41b203",
        "11ccd2d7350290fd5606c3d3b1785d5f12087b557cc0e63ceadf130aa55b1d4d",
        "960141f817f3689c0a77456c8b1bfc1745da5779aa2c6021fe697eadb3ff45bb",
        "d1906d310b2f3471aeb78ad50343b68cf42bf0677ae17460b3590a15d5450fa7",
    }
    expected_code = "af53963c8e8903a65bfe9fdc8fe5b553b12478596cd9397e12e98f920fc613a8"
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8-sig"))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name == "basis_identity_matches")
                 and not (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in {"RUN_START_DATE", "RUN_END_DATE", "RUN_TASK_IDS", "REBUILD_TASK_IDS"}
                     for target in node.targets))]
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()
    if saved.get("source_sha256") not in known_sources or digest != expected_code:
        return False
    return ({key: value for key, value in saved.items() if key != "source_sha256"}
            == {key: value for key, value in current.items() if key != "source_sha256"})


def build_basis(training_days, train_root, basis_path, random_seed, overwrite=False):
    dates = [date_from_day_folder(day).strftime("%Y-%m-%d") for day in training_days]
    inputs = [train_path_for_date(train_root, date_from_day_folder(day)) for day in training_days]
    source_tables = [p.with_name(p.name.replace("_traindata.npz", "_lineage.csv")) for p in inputs]
    identity = dict(science_identity(random_seed), training_dates=dates,
                    inputs={str(p.resolve()): file_hash(p) for p in inputs + source_tables})
    qa_path = basis_path.with_suffix(".qa.json")
    lineage_path = basis_path.with_suffix(".lineage.csv")
    counts_path = basis_path.with_suffix(".counts.csv")
    latitude_path = basis_path.with_suffix(".latitude.csv")
    if basis_path.exists() and not overwrite:
        if not qa_path.exists():
            raise ValueError(f"Basis 缺少来源记录，不能直接复用: {basis_path}")
        saved_qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if (not basis_identity_matches(saved_qa["identity"], identity) or saved_qa["basis_sha256"] != file_hash(basis_path)
                or saved_qa["lineage_sha256"] != file_hash(lineage_path)):
            raise ValueError(f"Basis 或训练来源已变化，请检查后决定是否 overwrite: {basis_path}")
        with np.load(basis_path, allow_pickle=False) as saved:
            if (float(saved["sza_limit"]) != SZA_LIMIT
                    or saved["training_dates"].tolist() != dates
                    or saved["V_all"].shape != (3, 72, 72)
                    or saved["S_all"].shape != (3, 72)
                    or saved["training_counts"].shape != (3,)):
                raise ValueError("Basis schema / SZA / 日期不一致")
            return saved["V_all"], saved["S_all"], saved["training_counts"]

    parts = {key: [] for key in NPZ_KEYS}
    source_rows = []
    for path, table in zip(inputs, source_tables):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(NPZ_KEYS):
                raise ValueError(f"训练文件六变量不完整: {path}")
            arrays = {key: np.asarray(data[key]) for key in NPZ_KEYS}
        n = len(arrays["traindata"])
        if arrays["traindata"].shape != (n, 72) or any(arrays[k].shape != (n,) for k in NPZ_KEYS[1:]):
            raise ValueError(f"训练文件行数/shape不一致: {path}")
        rows = read_csv(table)
        if len(rows) != n:
            raise ValueError(f"来源表行数不一致: {table}")
        if not np.isin(arrays["traincols"], [0, 1, 2]).all() or not np.isin(arrays["traintype"], [0, 1, 2]).all():
            raise ValueError(f"无效的 FOV 或类型: {path}")
        for i, row in enumerate(rows):
            if (int(row["daily_row_index"]) != i
                    or int(row["detector_column"]) != int(arrays["traincols"][i])):
                raise ValueError(f"来源表身份不一致: {table}, row={i}")
            source_rows.append(dict(row, source_npz=str(path.resolve())))
        for key in NPZ_KEYS:
            parts[key].append(arrays[key])
    arrays = {key: np.concatenate(values) for key, values in parts.items()}
    spectra = np.asarray(arrays["traindata"], dtype=np.float64)
    latitudes = np.asarray(arrays["trainlats"], dtype=np.float64)
    sza = np.asarray(arrays["trainsza"], dtype=np.float64)
    valid = np.isfinite(sza) & (sza < SZA_LIMIT) & np.all(np.isfinite(spectra), axis=1)
    if not np.isfinite(latitudes[valid]).all():
        raise ValueError("训练纬度含非有限值")
    V_all = np.full((3, 72, 72), np.nan)
    S_all = np.full((3, 72), np.nan)
    training_counts = np.zeros(3, dtype=np.int32)
    phase2 = date_from_day_folder(training_days[0]) >= PHASE_BOUNDARY
    token = date_from_day_folder(training_days[0]).strftime("%Y%m%d")
    count_rows, latitude_rows, selected_rows = [], [], []
    failed = False
    for fov_index in range(3):
        indices = np.flatnonzero(valid & (arrays["traincols"] == fov_index))
        types = arrays["traintype"][indices]
        lat = latitudes[indices]
        count = dict(Column=fov_index, FOV_Name=FOV_NAMES[fov_index], raw_total=len(indices),
                     raw_Soil=int(np.sum(types == 0)), raw_Cloud=int(np.sum(types == 1)),
                     raw_Snow=int(np.sum(types == 2)), sampled_total="", sampled_ge100="",
                     pca_total="", Status="SKIPPED_RAW_LT100")
        for lower in range(-90, 90, 10):
            within = (lat >= lower) & (lat < lower + 10)
            latitude_rows.append(dict(Column=fov_index, latitude_lower=lower,
                raw_Total=int(within.sum()), raw_Soil=int(np.sum(within & (types == 0))),
                raw_Cloud=int(np.sum(within & (types == 1))), raw_Snow=int(np.sum(within & (types == 2)))))
        if len(indices) < 100:
            failed = True
            count_rows.append(count)
            continue
        rng = np.random.default_rng(stable_seed(random_seed, f"{token}:{fov_index}"))
        chosen = balanced_training_indices(lat, types, phase2, rng)
        stage_b = indices[chosen]
        count.update(sampled_total=len(stage_b), sampled_ge100=len(stage_b) >= 100)
        for code, name in enumerate(("Soil", "Cloud", "Snow")):
            count["sampled_" + name] = int(np.sum(arrays["traintype"][stage_b] == code))
        stage_c = np.array([], dtype=np.int64)
        if len(stage_b) < 100:
            count["Status"] = "SKIPPED_SAMPLED_LT100"
            failed = True
        else:
            keep = pca_keep_mask(spectra[stage_b])
            stage_c = stage_b[keep]
            count["pca_total"] = len(stage_c)
            if len(stage_c) < SPECTRAL_CHANNELS:
                count["Status"] = "SKIPPED_PCA_LT72"
                failed = True
            else:
                _, singular_values, right_vectors = np.linalg.svd(spectra[stage_c], full_matrices=False)
                if right_vectors.shape != (72, 72):
                    raise ValueError("Unexpected SVD shape")
                V_all[fov_index] = right_vectors
                S_all[fov_index] = singular_values
                training_counts[fov_index] = len(stage_c)
                count["Status"] = "PASS"
        for stage, selected in (("StageB", stage_b), ("StageC", stage_c)):
            for order, index in enumerate(selected):
                selected_rows.append(dict(source_rows[index], stage=stage, stage_row=order,
                    Column=fov_index, traintype=int(arrays["traintype"][index]),
                    latitude=float(latitudes[index]), longitude=float(arrays["trainlons"][index]),
                    SZA=float(sza[index])))
        count_rows.append(count)
    write_csv(counts_path, count_rows)
    write_csv(latitude_path, latitude_rows)
    lineage_fields = ["daily_row_index", "source_file", "scan_index", "detector_column",
                      "raw_row_index", "source_npz", "stage", "stage_row", "Column",
                      "traintype", "latitude", "longitude", "SZA"]
    write_csv(lineage_path, selected_rows, lineage_fields)
    if failed:
        # 无论旧 basis 是否存在，本次均不得继续对应任务反演。
        raise InsufficientTraining(f"样本不足，已保存诊断: {counts_path}")
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = basis_path.with_name(basis_path.name + ".tmp.npz")
    np.savez(tmp, V_all=V_all, S_all=S_all, training_counts=training_counts,
             training_dates=np.array(dates), sza_limit=np.array(SZA_LIMIT))
    tmp.replace(basis_path)
    write_json(qa_path, dict(identity=identity, basis_sha256=file_hash(basis_path),
                            lineage_sha256=file_hash(lineage_path)))
    return V_all, S_all, training_counts


def retrieval_matrix(right_vectors: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    normalized = 2.0 * (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min()) - 1.0
    fluorescence_shape = np.exp(-((wavelength - 740.0) ** 2) / (2.0 * 10.0**2))
    matrix = np.zeros((SPECTRAL_CHANNELS, SVD_COMPONENTS + LEGENDRE_TERMS + 1))
    matrix[:, :SVD_COMPONENTS] = right_vectors[:SVD_COMPONENTS].T
    for order in range(1, LEGENDRE_TERMS + 1):
        matrix[:, SVD_COMPONENTS + order - 1] = eval_legendre(order, normalized) * right_vectors[0]
    matrix[:, -1] = fluorescence_shape
    return matrix


def retrieve_one_spectrum(matrix: np.ndarray, radiance: np.ndarray) -> tuple[float, float, float]:
    result, _, rank, _ = np.linalg.lstsq(matrix, radiance, rcond=None)
    if rank < matrix.shape[1]:
        raise np.linalg.LinAlgError("Rank-deficient SIF design matrix")
    rebuilt = matrix @ result
    residual = radiance - rebuilt
    degrees_of_freedom = len(radiance) - matrix.shape[1]
    residual_std = float(np.std(residual))
    mean_radiance = float(np.mean(radiance))
    if residual_std <= 0.0 or not np.isfinite(residual_std):
        raise ValueError("Invalid spectral residual standard deviation")
    snr = mean_radiance / residual_std
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_square = np.sum((rebuilt - radiance) ** 2 / (radiance / snr) ** 2)
    reduced_chi_square = float(chi_square / degrees_of_freedom)
    residual_variance = float(np.sum(residual**2) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.inv(matrix.T @ matrix)
    sif_variance = float(covariance[-1, -1])
    return float(result[-1]), reduced_chi_square, sif_variance


def float_array(variable, index) -> np.ndarray:
    values = np.ma.asarray(variable[index], dtype=np.float64)
    return np.ma.filled(values, np.nan)


def create_float_variable(dataset: nc.Dataset, name: str, values: np.ndarray, units: str, long_name: str):
    variable = dataset.createVariable(
        name, "f4", ("record",), zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )
    variable[:] = np.asarray(values, dtype=np.float32)
    variable.units = units
    variable.long_name = long_name
    return variable


def write_orbit_product(
    output_path: Path,
    source_path: Path,
    observation_date: datetime,
    arrays: dict[str, np.ndarray],
    training_counts: np.ndarray,
) -> None:
    """Write the confirmed compact public schema with minimal metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(arrays["sif_740"])
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with nc.Dataset(temporary_path, "w", format="NETCDF4") as dataset:
            dataset.createDimension("record", count)

            record = dataset.createVariable("record", "i4", ("record",))
            record[:] = np.arange(count, dtype=np.int32)
            record.long_name = "Valid retrieval record index"

            scan = dataset.createVariable(
                "scan_index", "i4", ("record",), zlib=True, complevel=4
            )
            scan[:] = arrays["scan_index"].astype(np.int32)
            scan.long_name = "Zero-based scanline index in the source GOME orbit"

            fov = dataset.createVariable(
                "scan_position", "i1", ("record",), zlib=True, complevel=4
            )
            fov[:] = arrays["detector_column"].astype(np.int8)
            fov.long_name = "GOME forward-scan ground-pixel position"
            fov.units = "1"
            fov.flag_values = np.array([0, 1, 2], dtype=np.int8)
            fov.flag_meanings = "east_pixel nadir_pixel west_pixel"

            create_float_variable(
                dataset,
                "sif_740",
                arrays["sif_740"],
                "mW m-2 sr-1 nm-1",
                "Solar-induced chlorophyll fluorescence at 740 nm",
            )
            create_float_variable(
                dataset,
                "sif_standard_error",
                arrays["sif_standard_error"],
                "mW m-2 sr-1 nm-1",
                "One-sigma theoretical SIF uncertainty",
            )
            create_float_variable(
                dataset,
                "reduced_chi_square",
                arrays["reduced_chi_square"],
                "1",
                "Reduced chi-square of the spectral fit",
            )
            create_float_variable(
                dataset, "latitude", arrays["latitude"], "degrees_north", "GOME footprint centre latitude"
            )
            create_float_variable(
                dataset, "longitude", arrays["longitude"], "degrees_east", "GOME footprint centre longitude"
            )
            create_float_variable(
                dataset,
                "mean_radiance",
                arrays["mean_radiance"],
                "mW m-2 sr-1 nm-1",
                "Mean radiance over 743-758 nm used in the retrieval",
            )
            create_float_variable(
                dataset, "solar_zenith_angle", arrays["solar_zenith_angle"], "degrees", "Solar zenith angle"
            )
            create_float_variable(
                dataset, "viewing_zenith_angle", arrays["viewing_zenith_angle"], "degrees", "Viewing zenith angle"
            )
            create_float_variable(
                dataset, "solar_azimuth_angle", arrays["solar_azimuth_angle"], "degrees", "Solar azimuth angle"
            )
            create_float_variable(
                dataset, "viewing_azimuth_angle", arrays["viewing_azimuth_angle"], "degrees", "Viewing azimuth angle"
            )
            create_float_variable(
                dataset, "cloud_fraction", arrays["cloud_fraction"], "1", "Source GOME footprint cloud fraction"
            )

            dataset.source_file = str(source_path.resolve())
            dataset.observation_date = observation_date.strftime("%Y-%m-%d")
            dataset.product_status = "pre-zero-offset-correction"
            dataset.experiment_name = "S2_SZA75"
            dataset.spectral_channel_count = SPECTRAL_CHANNELS
            dataset.svd_components = SVD_COMPONENTS
            dataset.legendre_polynomial_terms = LEGENDRE_TERMS
            dataset.training_spectra_after_pca = ",".join(str(int(n)) for n in training_counts)
            dataset.radiometric_degradation_correction = "Mali1 quadratic; C(t)=P(0)/P(t)"
            dataset.degradation_reference_date = REFERENCE_DATE.strftime("%Y-%m-%d")
            dataset.degradation_seasonal_terms_applied = "false"
            dataset.source = "ERS-2 GOME Level 1b Band 4"
            dataset.processing_level = "L2 orbit-level compact valid-retrieval product"
            dataset.spectral_window = "743-758 nm"
            dataset.production_quality_filter = (
                "land; latitude > -60; solar_zenith_angle < 75; viewing_zenith_angle < 60; "
                "-5 <= sif_740 <= 10; reduced_chi_square <= 1.2"
            )
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def nadir_geometry(band4):
    """Check variable shapes before indexing the first time/centre-angle sample."""
    geo = band4.groups['GEODATA'].variables
    variables = {
        'radiance': band4.groups['OBSERVATIONS'].variables['radiance'],
        'longitude': geo['longitude'], 'latitude': geo['latitude'],
        'SZA': geo['solar_zenith_angle'], 'VZA': geo['viewing_zenith_angle'],
        'cloud': band4.groups['CLOUDDATA'].variables['cloud_fraction'],
    }
    for key in ('solar_azimuth_angle', 'viewing_azimuth_angle'):
        if key in geo:
            variables[key] = geo[key]
    shapes = {name: tuple(var.shape) for name, var in variables.items()}
    lon = shapes['longitude']
    valid = len(lon) == 3 and lon[0] > 0 and lon[2] == 3
    footprint = lon[1:]
    for name, shape in shapes.items():
        if name == 'radiance':
            valid = valid and len(shape) == 4 and shape[0] > 0 and shape[1:3] == footprint
        elif name in ('longitude', 'latitude', 'cloud'):
            valid = valid and len(shape) == 3 and shape[0] > 0 and shape[1:] == footprint
        else:
            valid = valid and len(shape) == 4 and shape[0] > 0 and shape[1:3] == footprint and shape[3] > 1
    return bool(valid), shapes


def retrieve_orbit(
    source_path: Path,
    output_path: Path,
    observation_date: datetime,
    V_all: np.ndarray,
    training_counts: np.ndarray,
) -> dict[str, object]:
    with nc.Dataset(source_path) as dataset:
        mode = active_nadir_group(dataset)
        if mode is None:
            return {"status": "skipped_no_nadir", "records": 0}
        band4 = mode.groups["BAND_4"]
        geometry_ok, shapes = nadir_geometry(band4)
        if not geometry_ok:
            print(f"[SKIP_NONSTANDARD_GEOMETRY] {source_path} | shapes={shapes}", flush=True)
            return {"status": "skipped_nonstandard_geometry", "records": 0,
                    "geometry_shapes": str(shapes)}
        observations = band4.groups["OBSERVATIONS"]
        geodata = band4.groups["GEODATA"]
        clouddata = band4.groups["CLOUDDATA"]

        radiance = float_array(observations.variables["radiance"], 0)
        longitude = float_array(geodata.variables["longitude"], 0)
        latitude = float_array(geodata.variables["latitude"], 0)
        sza = float_array(geodata.variables["solar_zenith_angle"], (0, slice(None), slice(None), 1))
        vza = float_array(geodata.variables["viewing_zenith_angle"], (0, slice(None), slice(None), 1))
        saa = float_array(geodata.variables["solar_azimuth_angle"], (0, slice(None), slice(None), 1))
        vaa = float_array(geodata.variables["viewing_azimuth_angle"], (0, slice(None), slice(None), 1))
        cloud = float_array(clouddata.variables["cloud_fraction"], 0)

        wavelength_all = float_array(dataset.groups["CALIBRATION"].variables["wavelength"], (0, 3, slice(None)))
        channels_in_window = np.where(
            (wavelength_all >= START_WAVELENGTH_NM)
            & (wavelength_all <= END_WAVELENGTH_NM)
        )[0]
        if len(channels_in_window) < SPECTRAL_CHANNELS:
            raise ValueError(
                f"Expected at least 72 channels over 743-758 nm, found {len(channels_in_window)}"
            )
        # GOME wavelength calibration drifts slightly over the mission.  Some
        # 2008-2010 files contain 73 channel centres inside the nominal window.
        # Keep the production definition fixed at 72 consecutive channels,
        # starting at the first calibrated channel >= 743 nm.
        first_channel = int(channels_in_window[0])
        indices = np.arange(first_channel, first_channel + SPECTRAL_CHANNELS)
        if indices[-1] >= len(wavelength_all):
            raise ValueError("The 72-channel retrieval window exceeds the calibration array")
        selected_wavelengths = wavelength_all[indices]
        if (
            np.any(~np.isfinite(selected_wavelengths))
            or np.any(np.diff(selected_wavelengths) <= 0.0)
            or selected_wavelengths[0] < START_WAVELENGTH_NM
            or selected_wavelengths[-1] > END_WAVELENGTH_NM
        ):
            raise ValueError(
                "Invalid calibrated 72-channel wavelength selection: "
                f"{selected_wavelengths[0]:.6f}-{selected_wavelengths[-1]:.6f} nm"
            )
        wavelength = wavelength_all[indices]
        radiance = radiance[:, :, indices]
        converted = radiance * RADIANCE_CONVERSION_CONSTANT / wavelength[None, None, :]
        factors = correction_factors(observation_date)
        converted *= factors[None, :, None]

    matrices = [retrieval_matrix(V_all[index], wavelength) for index in range(FOV_COUNT)]
    shape = longitude.shape
    sif = np.full(shape, np.nan, dtype=np.float64)
    standard_error = np.full(shape, np.nan, dtype=np.float64)
    reduced_chi_square = np.full(shape, np.nan, dtype=np.float64)
    mean_radiance = np.full(shape, np.nan, dtype=np.float64)

    candidate_count = 0
    failure_count = 0
    for scan_index in range(shape[0]):
        for fov_index in range(shape[1]):
            if not (
                np.isfinite(longitude[scan_index, fov_index])
                and np.isfinite(latitude[scan_index, fov_index])
                and latitude[scan_index, fov_index] > LATITUDE_MIN
                and np.isfinite(sza[scan_index, fov_index])
                and sza[scan_index, fov_index] < SZA_LIMIT
                and np.isfinite(vza[scan_index, fov_index])
                and vza[scan_index, fov_index] < VZA_LIMIT
                and globe.is_land(latitude[scan_index, fov_index], longitude[scan_index, fov_index])
            ):
                continue
            spectrum = converted[scan_index, fov_index]
            if np.any(~np.isfinite(spectrum)):
                continue
            candidate_count += 1
            try:
                sif_value, chi_value, variance = retrieve_one_spectrum(matrices[fov_index], spectrum)
                if variance < 0.0 or not np.isfinite(variance):
                    raise ValueError("Invalid SIF variance")
                sif[scan_index, fov_index] = sif_value
                reduced_chi_square[scan_index, fov_index] = chi_value
                standard_error[scan_index, fov_index] = np.sqrt(variance)
                mean_radiance[scan_index, fov_index] = np.mean(spectrum)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                failure_count += 1

    valid = (
        np.isfinite(sif)
        & np.isfinite(reduced_chi_square)
        & (sif >= SIF_MIN)
        & (sif <= SIF_MAX)
        & (reduced_chi_square <= CHI_SQUARE_MAX)
    )
    scan_grid = np.broadcast_to(np.arange(shape[0])[:, None], shape)
    fov_grid = np.broadcast_to(np.arange(shape[1])[None, :], shape)
    factor_grid = np.broadcast_to(factors[None, :], shape)
    arrays = {
        "scan_index": scan_grid[valid],
        "detector_column": fov_grid[valid],
        "sif_740": sif[valid],
        "sif_standard_error": standard_error[valid],
        "reduced_chi_square": reduced_chi_square[valid],
        "latitude": latitude[valid],
        "longitude": longitude[valid],
        "mean_radiance": mean_radiance[valid],
        "solar_zenith_angle": sza[valid],
        "viewing_zenith_angle": vza[valid],
        "solar_azimuth_angle": saa[valid],
        "viewing_azimuth_angle": vaa[valid],
        "cloud_fraction": cloud[valid],
        "degradation_correction_factor": factor_grid[valid],
    }
    write_orbit_product(
        output_path,
        source_path,
        observation_date,
        arrays,
        training_counts,
    )
    return {
        "status": "written",
        "candidates": candidate_count,
        "fit_failures": failure_count,
        "records": int(np.sum(valid)),
    }


def orbit_output_path(output_root: Path, day: Path, source_path: Path) -> Path:
    value = date_from_day_folder(day)
    return output_root / str(value.year) / f"{value:%Y%m%d}" / f"{source_path.stem}_SIF.nc"


def write_manifest(output_root: Path, start: datetime, end: datetime, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"orbit_manifest_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def task_rows(tasks):
    rows = []
    for task_index, (training, inversion) in enumerate(tasks, start=1):
        rows.append({"Task_Index": str(task_index), "S2_Task_Index": str(task_index),
                     "Phase": "Phase 2" if date_from_day_folder(training[0]) >= PHASE_BOUNDARY else "Phase 1",
                     "Basis_Date": date_from_day_folder(inversion[0]).strftime("%Y-%m-%d"),
                     "Training_Dates": ";".join(date_from_day_folder(d).strftime("%Y-%m-%d") for d in training),
                     "Inversion_Dates": ";".join(date_from_day_folder(d).strftime("%Y-%m-%d") for d in inversion)})
    return rows


def task_definition(row):
    return tuple(row[k] for k in ("Phase", "Basis_Date", "Training_Dates", "Inversion_Dates"))


def fixed_batch_schedule(args):
    start, end = (datetime.fromisoformat(x) for x in (TASK_START_DATE, TASK_END_DATE))
    days = selected_day_folders(args.gome_root, args.train_root, start, end)
    tasks = production_tasks(days)
    schedule = task_rows(tasks)
    if not schedule:
        raise FileNotFoundError("总时期没有可用训练任务")
    path = args.output_root / f"S2_TASK_SCHEDULE_FIXED_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if path.exists():
        saved = read_csv(path)
        ids = [int(row["Task_Index"]) for row in saved]
        if (len(ids) != len(set(ids)) or any(i <= 0 for i in ids)
                or any(row["Task_Index"] != row["S2_Task_Index"] for row in saved)
                or any(row.get("Rebuild_Required", "0") not in ("0", "1") for row in saved)):
            raise ValueError("固定表编号或待重建标记无效")
        if [task_definition(row) for row in saved] != [task_definition(row) for row in schedule]:
            raise ValueError("可用训练日期/分组已变化；固定表不会自动重编号，请先核对新增或缺失日期")
        schedule = saved
    return path, schedule, tasks


def select_batch(schedule, args):
    start, end = datetime.fromisoformat(args.run_start), datetime.fromisoformat(args.run_end)
    if not datetime.fromisoformat(TASK_START_DATE) <= start <= end <= datetime.fromisoformat(TASK_END_DATE):
        raise ValueError("RUN 日期必须位于固定总时期内")
    ids = {int(r["Task_Index"]) for r in schedule}
    if set(args.task_ids) - ids:
        raise ValueError("指定了固定表中不存在的任务编号")
    # Select overlapping whole tasks; never truncate training or inversion groups.
    return [r for r in schedule if (not args.task_ids or int(r["Task_Index"]) in args.task_ids)
            and any(start <= datetime.fromisoformat(d) <= end for d in r["Inversion_Dates"].split(";"))]


def compatible_orbit_identity(saved, current, task, output_root):
    if saved == current:
        return True
    if {k: v for k, v in saved.items() if k != "Task_Index"} != {
            k: v for k, v in current.items() if k != "Task_Index"}:
        return False
    # An old number is valid only when an existing schedule proves the same group.
    for path in sorted(output_root.glob("S2_TASK_SCHEDULE_*.csv")):
        for old in read_csv(path):
            if (old["Task_Index"] == str(saved["Task_Index"])
                    and all(old[k] == task[k] for k in ("Basis_Date", "Training_Dates", "Inversion_Dates"))):
                return True
    return False


def verify_rebuild_archive(root):
    from archive_tasks_333_334_before_rebuild import guarded_path, BASIS_SUFFIXES
    root = guarded_path(root, root)
    archive_root = guarded_path(root / "_rebuild_backup_tasks_333_334", root)
    expected, verified = {}, set()
    manifests = sorted(archive_root.glob("*/archive_manifest.json"))
    if not manifests:
        raise ValueError("缺少重建备份清单")
    for path in manifests:
        guarded_path(path, root)
        data = json.loads(path.read_text(encoding="utf-8"))
        if Path(data["product_root"]).resolve() != root.resolve():
            raise ValueError("备份清单产品根目录不一致")
        for entry in data["files"]:
            source = guarded_path(Path(entry["source"]), root)
            relative = source.relative_to(root)
            parts = relative.parts
            allowed_basis = (len(parts) == 3 and parts[:2] == ("_basis", "2002")
                             and parts[2] in {f"SVD_Basis_{d}{s}" for d in ("20020501", "20020508") for s in BASIS_SUFFIXES})
            allowed_orbit = (len(parts) == 3 and parts[0] == "2002"
                             and parts[1] in {f"200205{d:02d}" for d in range(1, 15)} and parts[2].endswith("_SIF.nc"))
            allowed_qa = (len(parts) == 4 and parts[:2] == ("_qa", "2002")
                          and parts[2] in {f"200205{d:02d}" for d in range(1, 15)} and parts[3].endswith("_SIF.json"))
            if not (allowed_basis or allowed_orbit or allowed_qa):
                raise ValueError(f"备份范围异常: {source}")
            fingerprint = (entry["size"], entry["sha256"])
            if relative in expected and expected[relative] != fingerprint:
                raise ValueError(f"多次备份内容冲突: {relative}")
            expected[relative] = fingerprint
            if entry["status"] == "moved_verified":
                backup = guarded_path(Path(entry["backup"]), root)
                if backup != path.parent / relative or backup.stat().st_size != fingerprint[0] or file_hash(backup) != fingerprint[1]:
                    raise ValueError(f"备份内容校验失败: {backup}")
                verified.add(relative)
    if not expected or set(expected) != verified:
        raise ValueError("备份仍不完整；存在未成功移出的文件")
    for relative, (_, digest) in expected.items():
        source = guarded_path(root / relative, root)
        if source.exists() and file_hash(source) == digest:
            raise ValueError(f"旧文件仍在生产位置，拒绝覆盖: {source}")
    print(f"[REBUILD ARCHIVE VERIFIED] {len(verified)} files, {len(manifests)} manifests", flush=True)


def finish_rebuild_task(schedule_path, task):
    saved = read_csv(schedule_path)
    matches = [row for row in saved if row["Task_Index"] == task["Task_Index"]]
    if len(matches) != 1 or task_definition(matches[0]) != task_definition(task) or matches[0].get("Rebuild_Required") != "1":
        raise ValueError("重建期间任务表发生变化，保留标记并停止")
    matches[0]["Rebuild_Required"] = "0"
    write_csv(schedule_path, saved)
    task["Rebuild_Required"] = "0"
    print(f'[REBUILD COMPLETE] task={task["Task_Index"]}: Rebuild_Required=0', flush=True)


def main() -> None:
    args = parse_args()
    if args.max_orbits is not None and args.max_orbits <= 0:
        raise ValueError("max_orbits 必须为正数或 None")
    for path in (args.train_root, args.gome_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
        if args.output_root.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(args.output_root.resolve()):
            raise ValueError("输入与输出目录必须相互独立")
    forbidden = Path(r"F:\SIF740_GOME_N4_Mali1_quadratic_orbit_SZA70")
    if args.output_root.resolve().is_relative_to(forbidden.resolve()):
        raise ValueError("禁止写入旧 SZA70 baseline")
    schedule_path, schedule, tasks = fixed_batch_schedule(args)
    selected = select_batch(schedule, args)
    if not selected:
        raise ValueError("本次执行范围没有任务")
    print(f"固定任务={len(schedule)}，本次={len(selected)}；整组执行，不截断日期。")
    if args.dry_run:
        for task in selected:
            print(f'[DRY RUN] task={task["Task_Index"]}, basis={task["Basis_Date"]}, '
                  f'rebuild_required={task.get("Rebuild_Required", "0")}, inversion={task["Inversion_Dates"]}')
        return
    if set(REBUILD_TASK_IDS) - {333, 334}:
        raise ValueError("受控重建只支持333/334")
    rebuilding = [t for t in selected if int(t["Task_Index"]) in REBUILD_TASK_IDS and t.get("Rebuild_Required") == "1"]
    if rebuilding:
        if args.overwrite:
            raise ValueError("受控重建必须 overwrite=False")
        for task in rebuilding:
            expected_dates = ";".join(f"2002-05-{d:02d}" for d in (range(1, 8) if task["Task_Index"] == "333" else range(8, 15)))
            if task["Training_Dates"] != expected_dates or task["Inversion_Dates"] != expected_dates or task["Basis_Date"] != expected_dates[:10] or task.get("Replaces_Task_Index") != "323":
                raise ValueError("重建任务定义异常")
        verify_rebuild_archive(args.output_root)
    if not schedule_path.exists():
        write_csv(schedule_path, schedule)
    rows = []
    processed_orbits = 0
    skipped_tasks = 0
    blocked_tasks = 0
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    manifest_path = args.output_root / f"orbit_manifest_fixed_batch_{label}.csv"
    selected_ids = {int(r["Task_Index"]) for r in selected}
    for task_number, (task, (training_days, inversion_days)) in enumerate(zip(schedule, tasks), start=1):
        task_index = int(task["Task_Index"])
        if task_index not in selected_ids:
            continue
        first_date = date_from_day_folder(inversion_days[0])
        rebuilding_task = task in rebuilding
        if task.get("Rebuild_Required", "0") == "1" and not rebuilding_task:
            blocked_tasks += 1
            rows.append(dict(Task_Index=task_index, Basis_Date=f"{first_date:%Y-%m-%d}",
                             status="BLOCKED_TASK_REBUILD_REQUIRED",
                             error="旧分组已变化；保留旧产品，等待确认重建此任务"))
            write_csv(manifest_path, rows)
            print(f'[BLOCKED TASK] id={task_index}, basis={first_date:%Y-%m-%d}: 旧分组已变化，未反演', flush=True)
            continue
        basis_path = args.output_root / "_basis" / str(first_date.year) / f"SVD_Basis_{first_date:%Y%m%d}.npz"
        try:
            V_all, _, training_counts = build_basis(training_days, args.train_root, basis_path,
                                                     args.random_seed, args.overwrite)
        except InsufficientTraining as exc:
            skipped_tasks += 1
            rows.append(dict(Task_Index=task_index, Basis_Date=f"{first_date:%Y-%m-%d}",
                             status="SKIPPED_INSUFFICIENT_TRAINING", error=str(exc)))
            write_csv(manifest_path, rows)
            print(f"[SKIP TASK] {exc}", flush=True)
            continue
        print(f"[task {task_number}/{len(tasks)}; id={task_index}] basis={first_date:%Y-%m-%d}, "
              f"PCA后样本={training_counts.tolist()}", flush=True)
        basis_sha = file_hash(basis_path)
        task_row_start = len(rows)
        for day in inversion_days:
            observation_date = date_from_day_folder(day)
            daily_counts = {}
            daily_records = 0
            source_paths = sorted(day.rglob("*.nc"))
            if rebuilding_task and not source_paths:
                raise FileNotFoundError(f"重建日期没有L1B文件，保留标记: {day}")
            for source_path in source_paths:
                if args.max_orbits is not None and processed_orbits >= args.max_orbits:
                    write_csv(manifest_path, rows)
                    print(f'[DAY PARTIAL] {observation_date:%Y-%m-%d}: {daily_counts}; N={daily_records}', flush=True)
                    print(f"达到 max_orbits={args.max_orbits}，提前停止；并非整个范围完成。")
                    return
                output_path = orbit_output_path(args.output_root, day, source_path)
                qa_path = args.output_root / "_qa" / output_path.relative_to(args.output_root).with_suffix(".json")
                row = dict(Task_Index=task_index, Basis_Date=f"{first_date:%Y-%m-%d}",
                           date=f"{observation_date:%Y-%m-%d}", source_file=str(source_path),
                           output_file=str(output_path))
                try:
                    identity = dict(science_identity(args.random_seed), Task_Index=task_index,
                                    Basis_Date=row["Basis_Date"], basis_sha256=basis_sha,
                                    source_file=str(source_path.resolve()), source_sha256=file_hash(source_path))
                    if output_path.exists() and not args.overwrite:
                        if not qa_path.exists():
                            raise ValueError("已有轨道文件缺少来源 QA，不能直接跳过")
                        qa = json.loads(qa_path.read_text(encoding="utf-8"))
                        if (not compatible_orbit_identity(qa["identity"], identity, task, args.output_root)
                                or qa["output_sha256"] != file_hash(output_path)):
                            raise ValueError("已有轨道文件与当前来源/basis不一致")
                        row.update(status="verified_existing", records=qa["records"],
                                   Stored_Task_Index=qa["identity"]["Task_Index"],
                                   Fixed_Schedule=str(schedule_path))
                    else:
                        row.update(retrieve_orbit(source_path, output_path, observation_date, V_all, training_counts))
                        if row["status"] == "written":
                            with nc.Dataset(output_path, "r+") as dataset:
                                dataset.task_index = task_index
                                dataset.basis_date = row["Basis_Date"]
                                dataset.basis_sha256 = basis_sha
                            write_json(qa_path, dict(identity=identity, records=row["records"],
                                                    output_sha256=file_hash(output_path)))
                except (Exception, KeyboardInterrupt) as exc:
                    row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    write_csv(manifest_path, rows)
                    raise
                rows.append(row)
                processed_orbits += 1
                daily_counts[row["status"]] = daily_counts.get(row["status"], 0) + 1
                daily_records += row["records"]
            write_csv(manifest_path, rows)
            print(f'[DAY] {observation_date:%Y-%m-%d}: {daily_counts}; '
                  f'N={daily_records}; 累计轨道={processed_orbits}', flush=True)
        if rebuilding_task:
            completed = rows[task_row_start:]
            if not completed or any(r["status"] not in {"written", "verified_existing", "skipped_no_nadir", "skipped_nonstandard_geometry", "no_valid_pixels"} for r in completed):
                raise ValueError("重建存在未确认状态，保留待重建标记")
            finish_rebuild_task(schedule_path, task)
    print(f"所选范围处理结束：轨道={processed_orbits}，样本不足跳过任务={skipped_tasks}，待重建任务={blocked_tasks}。")
    if skipped_tasks or blocked_tasks:
        print("INCOMPLETE：存在样本不足或待重建任务，详见运行清单及 basis counts/latitude CSV。")
    else:
        print("本阶段输出为 pre-zero SIF，尚未应用零偏差校正。")


if __name__ == "__main__":
    main()
