"""Stage 05: apply frozen cosSZA_cosLat quadratic S2-SZA75 zero-bias coefficients (no interaction terms) to separate final orbit files."""
from __future__ import annotations

import json
import ast
import hashlib
import shutil
import time
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Any
import netCDF4 as nc
import numpy as np
import retrieve_gome_sif_orbit_Mali1_quadratic_S2_SZA75 as workflow

# 通常只修改本段；先完成 04，再运行本文件。
# 运行范围由04系数及冻结清单决定，不再按年份重新编号。
input_root = Path("outputs/pre_zero")
output_root = Path("outputs/zero_offset")
TASK_SCHEDULE = input_root / "S2_TASK_SCHEDULE_FIXED_19950701_20110630.csv"
BATCH_NAME = "full_19950701_20110630"  # 与04保持一致。
REGRESSION_MODEL = "SIF_zero/cos(SZA)=A+B*cos(SZA)+C*mean_radiance+D*cos(latitude)+E*cos(SZA)**2+F*mean_radiance**2+G*cos(latitude)**2"
coefficient_file = input_root / "_zero_offset_fit_cosSZA_cosLat" / BATCH_NAME / "coefficients.csv"
approved_manifest = coefficient_file.with_suffix(".manifest.json")
overwrite = False
dry_run = False
VERIFY_FROZEN_SOURCE_FILES = False  # 暂停启动时全量来源校验；恢复时改为True。
APPLY_START_DATE = "1995-07-01"
APPLY_END_DATE = "2011-06-30"


# Previous cosSZA_cosLat implementation: formula unchanged; full verification on adoption.
COMPATIBLE_APPLY_CODE = "1c46ff1f7034545bd116cd5de468589fc02bbab668515508515607c3459729a5"

FOV_NAMES = ("East", "Center", "West")
FORBIDDEN_ZERO_OFFSET_GLOBALS = {"zero_offset_correction_applied", "zero_offset_model",
                                "zero_offset_coefficient_sha256"}
REQUIRED_ORBIT_VARIABLES = (
    "sif_740",
    "sif_standard_error",
    "latitude",
    "longitude",
    "scan_position",
    "solar_zenith_angle",
    "viewing_zenith_angle",
    "solar_azimuth_angle",
    "viewing_azimuth_angle",
    "cloud_fraction",
    "mean_radiance",
    "reduced_chi_square",
)
NEW_VARIABLES = ("sif_bias", "sif_740_zero_offset_corrected")
def filled_values(variable: nc.Variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        fill = np.nan if np.issubdtype(variable.dtype, np.floating) else 0
        values = np.ma.filled(values, fill)
    return np.asarray(values)


def attr_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if np.issubdtype(left_array.dtype, np.number) and np.issubdtype(
        right_array.dtype, np.number
    ):
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def values_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def read_inputs_and_calculate(
    dataset: nc.Dataset,
    task_index: int,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    missing = set(REQUIRED_ORBIT_VARIABLES).difference(dataset.variables)
    if missing:
        raise ValueError(f"Missing required orbit variables: {sorted(missing)}")
    if any(name in dataset.variables for name in NEW_VARIABLES):
        raise ValueError("Input file already contains zero-offset variables")
    forbidden = FORBIDDEN_ZERO_OFFSET_GLOBALS.intersection(dataset.ncattrs())
    if forbidden:
        raise ValueError(f"Input file contains forbidden zero-offset globals: {sorted(forbidden)}")

    count = len(dataset.dimensions["record"])
    arrays = {
        name: filled_values(dataset.variables[name])
        for name in (
            "sif_740",
            "solar_zenith_angle",
            "mean_radiance",
            "latitude",
            "scan_position",
        )
    }
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError("Required orbit variables are not aligned to the record dimension")
    if not np.all(np.isin(arrays["scan_position"], (0, 1, 2))):
        raise ValueError("At least one record has scan_position outside 0,1,2")
    fov = arrays["scan_position"].astype(np.int64, copy=False)
    for name in ("sif_740", "solar_zenith_angle", "mean_radiance", "latitude"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"At least one record has non-finite {name}")

    per_record = coefficients[task_index, fov]
    if per_record.shape != (count, 7) or not np.all(np.isfinite(per_record)):
        raise ValueError("At least one record failed exact FOV coefficient mapping")
    sza = arrays["solar_zenith_angle"].astype(np.float64)
    radiance = arrays["mean_radiance"].astype(np.float64)
    latitude = arrays["latitude"].astype(np.float64)
    cosine = np.cos(np.deg2rad(sza))
    latitude_predictor = np.cos(np.deg2rad(latitude))
    bias_normalized = (
        per_record[:, 0]
        + per_record[:, 1] * cosine
        + per_record[:, 2] * radiance
        + per_record[:, 3] * latitude_predictor
        + per_record[:, 4] * cosine**2
        + per_record[:, 5] * radiance**2
        + per_record[:, 6] * latitude_predictor**2
    )
    bias_float32 = np.asarray(
        cosine * bias_normalized,
        dtype=np.float32,
    )
    sif_float32 = np.asarray(arrays["sif_740"], dtype=np.float32)
    corrected_float32 = np.asarray(sif_float32 - bias_float32, dtype=np.float32)
    if not np.all(np.isfinite(bias_float32)) or not np.all(np.isfinite(corrected_float32)):
        raise ValueError("Zero-offset calculation produced non-finite values")
    return bias_float32, corrected_float32


def create_zero_offset_variable(
    dataset: nc.Dataset,
    name: str,
    values: np.ndarray,
    long_name: str,
    description: str,
) -> None:
    variable = dataset.createVariable(
        name,
        "f4",
        ("record",),
        zlib=True,
        complevel=4,
        fill_value=np.float32(np.nan),
    )
    variable[:] = values
    variable.long_name = long_name
    variable.description = description
    variable.units = "mW m-2 sr-1 nm-1"



def load_coefficients():
    started = time.monotonic()
    print("[校验开始] 读取04冻结清单及系数，尚未开始apply。", flush=True)
    frozen = json.loads(approved_manifest.read_text(encoding="utf-8"))
    if frozen["settings"].get("regression_model") != REGRESSION_MODEL:
        raise ValueError("冻结系数模型与当前apply不一致")
    actual_hash = workflow.file_hash(coefficient_file)
    if frozen["coefficient_sha256"] != actual_hash:
        raise ValueError("系数 SHA256 与04冻结清单不一致，停止apply")
    if Path(frozen["coefficient_file"]).resolve() != coefficient_file.resolve():
        raise ValueError("冻结清单对应不同的系数文件")
    if VERIFY_FROZEN_SOURCE_FILES:
        total = sum(len(frozen[section]) for section in ("sources", "artifacts"))
        checked = 0
        last_report = time.monotonic()
        print(f"[文件校验] 0/{total}，逐文件检查SHA256。", flush=True)
        for section in ("sources", "artifacts"):
            for name, sha in frozen[section].items():
                if workflow.file_hash(Path(name)) != sha:
                    raise ValueError(f"04冻结后的来源或样本记录已变化: {name}")
                checked += 1
                now = time.monotonic()
                if checked % 500 == 0 or checked == total or now - last_report >= 10:
                    print(f"[文件校验] {checked}/{total}，已耗时{now - started:.1f}秒；刚完成: {Path(name).name}", flush=True)
                    last_report = now
        print("[文件校验完成]", flush=True)
    else:
        print("[校验已暂停] 不检查04来源文件及样本记录是否发生变化；系数和任务表校验仍保留。", flush=True)
    print("[任务校验] 正在检查固定任务与FOV系数映射。", flush=True)
    schedule_path = TASK_SCHEDULE
    if Path(frozen["schedule_file"]).resolve() != schedule_path.resolve():
        raise ValueError("固定任务表路径与04冻结清单不一致")
    if frozen["sources"].get(str(schedule_path.resolve())) != workflow.file_hash(schedule_path):
        raise ValueError("任务表哈希不一致")
    schedule = workflow.read_csv(schedule_path)
    selected = frozen["selected_task_ids"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("04冻结任务编号为空或重复")
    schedule = [t for t in schedule if int(t["Task_Index"]) in selected]
    if len(schedule) != len(selected) or schedule != frozen["selected_tasks"]:
        raise ValueError("固定任务定义与04冻结任务不一致")
    tasks, dates = {}, {}
    for task in schedule:
        index = int(task["Task_Index"])
        if index <= 0 or index in tasks or int(task["S2_Task_Index"]) != index:
            raise ValueError("任务编号无效或重复")
        tasks[index] = task
        inversion = task["Inversion_Dates"].split(";")
        if task["Basis_Date"] != inversion[0]:
            raise ValueError("Basis_Date与任务不一致")
        for date in inversion:
            value = datetime.strptime(date, "%Y-%m-%d")
            if date in dates:
                raise ValueError("日期映射重复")
            dates[date] = index
    if not tasks:
        raise ValueError("任务表为空")
    rows = workflow.read_csv(coefficient_file)
    if len(rows) != len(tasks) * 3 or frozen["expected_rows"] != len(rows):
        raise ValueError("系数行数与任务表不一致")
    coefficients = np.full((max(tasks) + 1, 3, 7), np.nan)
    seen, blocked, good_rows = set(), set(), {}
    for row in rows:
        if row.get("Regression_Model") != REGRESSION_MODEL:
            raise ValueError("系数行模型与当前apply不一致")
        task, col = int(row["Task_Index"]), int(row["Column"])
        if task not in tasks or col not in (0, 1, 2) or (task, col) in seen:
            raise ValueError("系数task/FOV重复或无效")
        seen.add((task, col))
        if (row["FOV_Name"] != FOV_NAMES[col]
                or row["Basis_Date"] != tasks[task]["Basis_Date"]
                or row["Training_Dates"] != tasks[task]["Training_Dates"]
                or row["Inversion_Dates"] != tasks[task]["Inversion_Dates"]):
            raise ValueError("系数日期/FOV与固定任务不一致")
        if row["Status"] != "ok":
            blocked.add(task)
            continue
        if int(row["Fit_Sample_Size"]) < 100 or int(row["Design_Matrix_Rank"]) != 7:
            raise ValueError("标为ok的系数未满足正式拟合门槛")
        values = np.array([float(row[k]) for k in ("A_Intercept", "B_CosSZA", "C_Radiance", "D_CosLatitude", "E_CosSZA2", "F_Radiance2", "G_CosLatitude2")])
        if not np.isfinite(values).all():
            raise ValueError("系数含非有限值")
        coefficients[task, col] = values
        good_rows[(task, col)] = row
    if seen != {(task, col) for task in tasks for col in range(3)}:
        raise ValueError("缺少FOV系数")
    approved = set(tasks) - blocked
    if approved != set(frozen["approved_tasks"]) or blocked != set(frozen["blocked_tasks"]):
        raise ValueError("任务就绪状态与04冻结清单不一致")
    for task in approved:
        if tasks[task].get("Rebuild_Required", "0") != "0":
            raise ValueError("待重建任务禁止apply")
        hashes = {good_rows[(task, col)]["Basis_SHA256"] for col in range(3)}
        if len(hashes) != 1:
            raise ValueError("三个FOV的basis哈希不一致")
        tasks[task]["Basis_SHA256"] = hashes.pop()
    print(f"[校验完成] 任务={len(tasks)}，禁止apply={len(blocked)}，总耗时{time.monotonic() - started:.1f}秒。", flush=True)
    return coefficients, tasks, dates, blocked, actual_hash


def validate_output(source_path, output_path, task, coefficients, metadata,
                    expected_values=None, compare_original_values=True):
    with nc.Dataset(source_path) as source, nc.Dataset(output_path) as output:
        dims = lambda ds: {k: (len(v), v.isunlimited()) for k, v in ds.dimensions.items()}
        if dims(source) != dims(output) or set(output.variables) != set(source.variables) | set(NEW_VARIABLES):
            raise ValueError("输出记录数或变量集合不一致")
        # 检查实际存储值，保留填充值和原变量metadata。
        for name, original in source.variables.items():
            copied = output.variables[name]
            original.set_auto_maskandscale(False)
            copied.set_auto_maskandscale(False)
            if original.dtype != copied.dtype or original.dimensions != copied.dimensions or (compare_original_values and not values_equal(original[:], copied[:])):
                raise ValueError(f"原变量被改变: {name}")
            if set(original.ncattrs()) != set(copied.ncattrs()) or any(
                    not attr_equal(original.getncattr(k), copied.getncattr(k)) for k in original.ncattrs()):
                raise ValueError(f"原变量metadata被改变: {name}")
            original.set_auto_maskandscale(True)
            copied.set_auto_maskandscale(True)
        if set(output.ncattrs()) != set(source.ncattrs()) | set(metadata):
            raise ValueError("全局metadata存在非预期增删")
        for key in output.ncattrs():
            expected = metadata[key] if key in metadata else source.getncattr(key)
            if not attr_equal(output.getncattr(key), expected):
                raise ValueError("输出metadata不一致: " + key)
        expected_bias, expected_corrected = (
            read_inputs_and_calculate(source, task, coefficients)
            if expected_values is None else expected_values
        )
        for name, expected in zip(NEW_VARIABLES, (expected_bias, expected_corrected)):
            var = output.variables[name]
            if var.dtype != np.dtype("float32") or var.dimensions != ("record",):
                raise ValueError("新增变量schema不一致")
            if not np.array_equal(filled_values(var), expected, equal_nan=True):
                raise ValueError("float32偏差公式/相减恒等式不一致")
        return len(source.dimensions["record"])


@lru_cache(maxsize=1)
def apply_code_hash(source_text):
    tree = ast.parse(source_text)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
            "APPLY_START_DATE", "APPLY_END_DATE", "VERIFY_FROZEN_SOURCE_FILES"
        } for t in node.targets))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def current_apply_code_hash():
    return apply_code_hash(Path(__file__).read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=256)
def compatible_task_mapping(root, old_id, new_id, basis_date, training_dates, inversion_dates):
    return workflow.compatible_orbit_identity(
        {"Task_Index": old_id}, {"Task_Index": new_id},
        dict(Basis_Date=basis_date, Training_Dates=training_dates, Inversion_Dates=inversion_dates),
        root)


def coefficient_stamp():
    info = coefficient_file.stat()
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns


def file_stamp(path):
    info = path.stat()
    return f"{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}"


def completed_row(output_path, count, source_stamp, qa_stamp, status):
    return dict(status=status, records=count, output_file=str(output_path),
                source_stamp=source_stamp, qa_stamp=qa_stamp,
                output_stamp=file_stamp(output_path))


def apply_one(source_path, task_id, task, date, coefficients, coefficient_sha, previous=None):
    output_path = output_root / source_path.relative_to(input_root)
    source_stamp = file_stamp(source_path)
    with nc.Dataset(source_path) as source:
        stored_task_id = int(source.getncattr("task_index"))
        if (source.getncattr("basis_date") != task["Basis_Date"]
                or source.getncattr("observation_date") != date
                or source.getncattr("basis_sha256") != task["Basis_SHA256"]):
            raise ValueError("轨道task/date/basis与系数不一致")
        if source.getncattr("product_status") != "pre-zero-offset-correction":
            raise ValueError("输入不是pre-zero产品，拒绝重复校正")
    original_qa = input_root / "_qa" / source_path.relative_to(input_root).with_suffix(".json")
    qa_stamp = file_stamp(original_qa)
    qa = json.loads(original_qa.read_text(encoding="utf-8"))
    saved_identity = qa["identity"]
    if int(saved_identity["Task_Index"]) != stored_task_id:
        raise ValueError("轨道编号与03 QA不一致")
    if stored_task_id != task_id and not compatible_task_mapping(
            input_root, stored_task_id, task_id, task["Basis_Date"],
            task["Training_Dates"], task["Inversion_Dates"]):
        raise ValueError("旧轨道编号无法通过历史任务表映射到当前固定分组")
    if qa["identity"]["basis_sha256"] != task["Basis_SHA256"]:
        raise ValueError("03 QA的basis来源不一致")
    # Only successful, unchanged manifest entries qualify for lightweight resume.
    if (not overwrite and output_path.exists() and previous
            and previous.get("status") in ("written", "verified_existing", "skipped_existing")
            and previous.get("coefficient_sha256") == coefficient_sha
            and previous.get("date") == date
            and str(previous.get("Task_Index")) == str(task_id)
            and previous.get("output_file") == str(output_path)
            and previous.get("source_stamp") == source_stamp
            and previous.get("qa_stamp") == qa_stamp
            and previous.get("output_stamp") == file_stamp(output_path)):
        with nc.Dataset(output_path) as existing, nc.Dataset(source_path) as source:
            expected = dict(product_status="zero-offset-corrected",
                            zero_offset_correction_applied="true",
                            zero_offset_model=REGRESSION_MODEL,
                            zero_offset_coefficient_sha256=coefficient_sha,
                            zero_offset_coefficient_file=str(coefficient_file.resolve()),
                            zero_offset_input_sha256=qa["output_sha256"],
                            zero_offset_task_index=task_id,
                            zero_offset_basis_date=task["Basis_Date"])
            if any(k not in existing.ncattrs() or not attr_equal(existing.getncattr(k), v)
                   for k, v in expected.items()):
                raise ValueError("已有输出模型/来源不一致")
            if existing.getncattr("zero_offset_apply_source_sha256") not in (
                    current_apply_code_hash(), COMPATIBLE_APPLY_CODE):
                raise ValueError("已有输出源码版本不一致")
            count = len(source.dimensions["record"])
            if (len(existing.dimensions["record"]) != count
                    or int(previous["records"]) != count
                    or set(existing.variables) != set(source.variables) | set(NEW_VARIABLES)):
                raise ValueError("已有输出记录数/变量不一致")
            for name in NEW_VARIABLES:
                var = existing.variables[name]
                if var.dtype != np.dtype("float32") or var.dimensions != ("record",):
                    raise ValueError("已有输出新增变量结构不一致")
        return completed_row(output_path, count, source_stamp, qa_stamp, "skipped_existing")
    source_sha = workflow.file_hash(source_path)
    if qa["output_sha256"] != source_sha or qa["identity"]["basis_sha256"] != task["Basis_SHA256"]:
        raise ValueError("03轨道文件或basis来源不一致")
    metadata = dict(product_status="zero-offset-corrected", zero_offset_correction_applied="true",
                    zero_offset_model=REGRESSION_MODEL,
                    zero_offset_coefficient_sha256=coefficient_sha,
                    zero_offset_coefficient_file=str(coefficient_file.resolve()),
                    zero_offset_input_sha256=source_sha,
                    zero_offset_apply_source_sha256=current_apply_code_hash(),
                    zero_offset_task_index=task_id, zero_offset_basis_date=task["Basis_Date"])
    if output_path.exists() and not overwrite:
        with nc.Dataset(output_path) as existing:
            stored_code = existing.getncattr("zero_offset_apply_source_sha256")
        if stored_code not in (metadata["zero_offset_apply_source_sha256"], COMPATIBLE_APPLY_CODE):
            raise ValueError("已有产品的apply源码版本与当前模型不一致")
        metadata["zero_offset_apply_source_sha256"] = stored_code
        count = validate_output(source_path, output_path, task_id, coefficients, metadata)
        if file_stamp(source_path) != source_stamp or file_stamp(original_qa) != qa_stamp:
            raise ValueError("验证期间输入或QA发生变化")
        return completed_row(output_path, count, source_stamp, qa_stamp, "verified_existing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.nc")
    try:
        shutil.copy2(source_path, temporary)
        with nc.Dataset(temporary, "r+") as output:
            bias, corrected = read_inputs_and_calculate(output, task_id, coefficients)
            create_zero_offset_variable(output, "sif_bias", bias, "Estimated SIF zero-offset bias",
                                        "Bias estimated with frozen task/FOV Stage C coefficients: " + REGRESSION_MODEL)
            create_zero_offset_variable(output, "sif_740_zero_offset_corrected", corrected,
                                        "Zero-offset-corrected SIF at 740 nm",
                                        "SIF after Mali1 quadratic degradation and zero-offset corrections")
            output.setncatts(metadata)
        count = validate_output(source_path, temporary, task_id, coefficients, metadata,
                                expected_values=(bias, corrected), compare_original_values=False)
        if file_stamp(source_path) != source_stamp or file_stamp(original_qa) != qa_stamp:
            raise ValueError("apply期间输入或QA发生变化")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return completed_row(output_path, count, source_stamp, qa_stamp, "written")


def manifest_key(row):
    source = row.get("source_file")
    return ("orbit", str(Path(source).resolve())) if source else ("day", row["date"], str(row["Task_Index"]))


def main():
    if not BATCH_NAME or Path(BATCH_NAME).name != BATCH_NAME:
        raise ValueError("BATCH_NAME必须是单个目录名")
    before, after = input_root.resolve(), output_root.resolve()
    if before.is_relative_to(after) or after.is_relative_to(before):
        raise ValueError("pre-zero与final必须是独立目录")
    forbidden = Path(r"F:\SIF740_GOME_N4_Mali1_quadratic_orbit_SZA70_zero_offset")
    if after.is_relative_to(forbidden.resolve()):
        raise ValueError("禁止写入旧baseline")
    start = datetime.strptime(APPLY_START_DATE, "%Y-%m-%d")
    end = datetime.strptime(APPLY_END_DATE, "%Y-%m-%d")
    if start > end:
        raise ValueError("APPLY_START_DATE不能晚于APPLY_END_DATE")
    initial_stamp = coefficient_stamp()
    coefficients, tasks, dates, blocked, frozen_sha = load_coefficients()
    if coefficient_stamp() != initial_stamp:
        raise ValueError("读取期间系数文件改变")
    dates = {date: task for date, task in dates.items()
             if start <= datetime.strptime(date, "%Y-%m-%d") <= end}
    if not dates:
        raise ValueError("所选日期没有可处理的固定任务日期")
    if dry_run:
        print(f"[DRY RUN] {len(tasks)} tasks; blocked={sorted(blocked)}; output={output_root}")
        return
    current_apply_code_hash.cache_clear()
    current_apply_code_hash()
    print(f"[APPLY开始] {APPLY_START_DATE}至{APPLY_END_DATE}，共{len(dates)}天；此前轨道不再遍历。", flush=True)
    label = BATCH_NAME
    manifest_path = output_root / f"zero_offset_manifest_{label}.csv"
    records = {}
    if manifest_path.exists():
        for row in workflow.read_csv(manifest_path):
            if row.get("coefficient_sha256") and row["coefficient_sha256"] != frozen_sha:
                raise ValueError("已有运行清单属于不同系数，停止续跑")
            records[manifest_key(row)] = row
    active_row = None
    last_manifest_save = time.monotonic()
    try:
        for date, task_id in sorted(dates.items()):
            day_started = time.monotonic()
            daily_counts, daily_records = {}, 0
            if coefficient_stamp() != initial_stamp:
                raise ValueError("运行期间系数文件改变")
            # Revalidate historical task mappings once per day, not once per orbit.
            compatible_task_mapping.cache_clear()
            if task_id in blocked:
                row = dict(date=date, Task_Index=task_id, status="BLOCKED_FIT_TASK")
                records[manifest_key(row)] = row
                daily_counts["BLOCKED_FIT_TASK"] = 1
            else:
                value = datetime.strptime(date, "%Y-%m-%d")
                files = sorted((input_root / str(value.year) / f"{value:%Y%m%d}").glob("*_SIF.nc"))
                if not files:
                    row = dict(date=date, Task_Index=task_id, status="MISSING_PREZERO")
                    records[manifest_key(row)] = row
                    daily_counts["MISSING_PREZERO"] = 1
                else:
                    records.pop(("day", date, str(task_id)), None)
                for source in files:
                    active_row = dict(date=date, Task_Index=task_id, source_file=str(source),
                                      coefficient_sha256=frozen_sha)
                    previous = records.get(manifest_key(active_row))
                    active_row.update(apply_one(source, task_id, tasks[task_id], date, coefficients,
                                               frozen_sha, previous=previous))
                    records[manifest_key(active_row)] = active_row
                    status = active_row["status"]
                    daily_counts[status] = daily_counts.get(status, 0) + 1
                    daily_records += active_row["records"]
                    active_row = None
            if time.monotonic() - last_manifest_save >= 60:
                workflow.write_csv(manifest_path, list(records.values()))
                last_manifest_save = time.monotonic()
            print(f"[DAY] {date}: {daily_counts}; N={daily_records}; "
                  f"耗时={time.monotonic() - day_started:.1f}秒", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        if active_row is not None:
            active_row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error=f"{type(exc).__name__}: {exc}")
            records[manifest_key(active_row)] = active_row
        workflow.write_csv(manifest_path, list(records.values()))
        raise
    if coefficient_stamp() != initial_stamp or workflow.file_hash(coefficient_file) != frozen_sha:
        raise ValueError("结束时系数文件发生变化")
    workflow.write_csv(manifest_path, list(records.values()))
    incomplete = any(r["status"] not in ("written", "verified_existing", "skipped_existing") for r in records.values())
    print(f'{"INCOMPLETE" if incomplete else "完成"}: {output_root}')
    print("最终变量: sif_740_zero_offset_corrected；原sif_740保持不变。")


if __name__ == "__main__":
    main()
