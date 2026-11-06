"""Create and verify the separate GOME-LSIF v1.0 public product.

Requires numpy, netCDF4 and tqdm. Source products are opened read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from tqdm.auto import tqdm


# ==================== VSCode 运行配置 ====================
INPUT_ROOT = Path("outputs/zero_offset")
OUTPUT_ROOT = Path("outputs/GOME_LSIF_public_v1_0")
# ========================================================

SOURCE_MANIFEST_NAME = "zero_offset_manifest_full_19950701_20110630.csv"
SCRIPT_PATH = Path(__file__).resolve()
UNITS = "mW m-2 sr-1 nm-1"
FILTER = (
    "land; latitude > -60; solar_zenith_angle < 75; viewing_zenith_angle < 60; "
    "-5 <= sif_740 <= 5; reduced_chi_square <= 1.2; 25 <= mean_radiance <= 200"
)
EXPECTED_VARIABLES = {
    "record", "scan_index", "scan_position", "sif_740", "sif_standard_error",
    "reduced_chi_square", "latitude", "longitude", "mean_radiance",
    "solar_zenith_angle", "viewing_zenith_angle", "solar_azimuth_angle",
    "viewing_azimuth_angle", "cloud_fraction", "sif_bias",
    "sif_740_zero_offset_corrected",
}
MANIFEST_FIELDS = [
    "file", "date", "status", "input_records", "retained_records",
    "removed_records", "failed_sif", "failed_radiance", "failed_both",
    "invalid_cloud_retained", "source_sha256", "output_sha256", "error",
]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def inventory(root):
    """核对真实文件与原生产清单，避免静默遗漏或处理清单外文件。"""
    with (root / SOURCE_MANIFEST_NAME).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = {}
    for row in rows:
        relative = Path(row["output_file"]).resolve().relative_to(root)
        if relative in entries:
            raise ValueError(f"原清单有重复路径：{relative}")
        if not re.fullmatch(r"\d{4}/\d{8}/[^/]+\.nc", relative.as_posix()):
            raise ValueError(f"非预期轨道路径：{relative}")
        entries[relative] = row
    actual = set()
    years = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}", p.name))
    for year in tqdm(years, desc="核对文件目录", unit="year"):
        actual.update(p.relative_to(root) for p in year.rglob("*.nc"))
    missing, extra = set(entries) - actual, actual - set(entries)
    if missing or extra:
        raise ValueError(
            f"目录与生产清单不一致：缺少 {len(missing)} 个，多出 {len(extra)} 个；"
            f"示例：{list(sorted(missing))[:2]} / {list(sorted(extra))[:2]}"
        )
    if not entries:
        raise ValueError("未发现轨道文件")
    return sorted(entries.items())


def variable_metadata(name, original):
    # 仅继承单位和缺失值；方法说明等源属性不自动带入共享产品。
    attrs = {key: original.getncattr(key) for key in ("units", "_FillValue")
             if key in original.ncattrs()}
    long_names = {
        "record": "Retained source-product record index",
        "scan_index": "Zero-based source-orbit scanline index",
        "scan_position": "GOME forward-scan ground-pixel position",
        "sif_740": "SIF at 740 nm before zero-offset correction",
        "sif_standard_error": "One-sigma theoretical SIF retrieval uncertainty",
        "sif_bias": "Estimated SIF zero-offset bias",
        "sif_740_zero_offset_corrected": "Zero-offset-corrected SIF at 740 nm",
        "latitude": "GOME footprint centre latitude",
        "longitude": "GOME footprint centre longitude",
        "mean_radiance": "Mean degradation-corrected radiance over 743-758 nm",
        "reduced_chi_square": "Reduced chi-square of the spectral fit",
        "cloud_fraction": "GOME footprint cloud fraction",
        "solar_zenith_angle": "Solar zenith angle",
        "viewing_zenith_angle": "Viewing zenith angle",
        "solar_azimuth_angle": "Solar azimuth angle",
        "viewing_azimuth_angle": "Viewing azimuth angle",
    }
    attrs["long_name"] = long_names[name]
    if name not in {"record", "latitude", "longitude"}:
        attrs["coordinates"] = "latitude longitude"
    if name in {"latitude", "longitude", "solar_zenith_angle"}:
        attrs["standard_name"] = name
    if name == "viewing_zenith_angle":
        attrs["standard_name"] = "sensor_zenith_angle"
    if name == "scan_position":
        attrs["flag_values"] = np.array([0, 1, 2], dtype=original.dtype)
        attrs["flag_meanings"] = "east_pixel nadir_pixel west_pixel"
    if name in {"sif_740", "sif_740_zero_offset_corrected"}:
        attrs["ancillary_variables"] = "sif_standard_error reduced_chi_square"
    if name == "sif_740_zero_offset_corrected":
        attrs["ancillary_variables"] += " sif_bias"
    if name in {"sif_740", "sif_standard_error", "sif_bias", "sif_740_zero_offset_corrected", "mean_radiance"}:
        attrs["units"] = UNITS
    return attrs


def global_metadata(source):
    # 固定9项全局属性；哈希、运行信息和方法细节保留在清单、报告及指南中。
    return dict(
        title="GOME-LSIF solar-induced chlorophyll fluorescence at 740 nm",
        product_version="1.0",
        source="ERS-2 GOME Level 1b Band 4",
        source_file=Path(str(source.getncattr("source_file")).replace("\\", "/")).name,
        processing_level="L2",
        observation_date=source.getncattr("observation_date"),
        spectral_window="743-758 nm",
        production_quality_filter=FILTER,
        license="CC BY 4.0",
    )


def equal_attribute(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def validate_output(path, source, arrays, keep, expected_attrs):
    data = path.read_bytes()
    with nc.Dataset(path.name, memory=data) as out:
        out.set_auto_maskandscale(False)
        if set(out.variables) != EXPECTED_VARIABLES or set(out.dimensions) != {"record"}:
            raise ValueError("输出变量集合或维度不一致")
        if len(out.dimensions["record"]) != int(keep.sum()):
            raise ValueError("输出记录数不一致")
        for name, original in source.variables.items():
            variable = out[name]
            expected = np.ascontiguousarray(arrays[name][keep])
            observed = np.ascontiguousarray(variable[:])
            if (variable.dtype != original.dtype or variable.dimensions != ("record",)
                    or observed.shape != expected.shape or observed.tobytes() != expected.tobytes()):
                raise ValueError(f"变量 {name} 的类型、顺序或保留值发生变化")
            attrs = variable_metadata(name, original)
            if set(variable.ncattrs()) != set(attrs):
                raise ValueError(f"变量 {name} 属性集合不一致")
            for key, value in attrs.items():
                if not equal_attribute(variable.getncattr(key), value):
                    raise ValueError(f"变量属性不一致：{name}:{key}")
        if set(out.ncattrs()) != set(expected_attrs):
            raise ValueError("全局属性集合不一致")
        for key, value in expected_attrs.items():
            if not equal_attribute(out.getncattr(key), value):
                raise ValueError(f"全局属性不一致：{key}")
    return sha256(data)


def convert_file(source_path, target_path, relative, entry):
    stamp = source_path.stat()
    blob = source_path.read_bytes()
    source_hash = sha256(blob)
    with nc.Dataset(source_path.name, memory=blob) as source:
        source.set_auto_maskandscale(False)
        if set(source.variables) != EXPECTED_VARIABLES or set(source.dimensions) != {"record"}:
            raise ValueError("输入结构不同于已核对的16变量产品，请先核查")
        if any(v.dimensions != ("record",) for v in source.variables.values()):
            raise ValueError("存在不沿 record 维度存储的变量")
        arrays = {name: np.asarray(v[:]) for name, v in source.variables.items()}
        n = len(arrays["record"])
        if n != int(entry["records"]):
            raise ValueError("输入记录数与原生产清单不一致")
        sif, radiance = arrays["sif_740"], arrays["mean_radiance"]
        good_sif = np.isfinite(sif) & (sif >= -5) & (sif <= 5)
        good_radiance = np.isfinite(radiance) & (radiance >= 25) & (radiance <= 200)
        keep = good_sif & good_radiance  # 云量、校正后 SIF 均不加入筛选。
        cloud = arrays["cloud_fraction"]
        invalid_cloud = ~np.isfinite(cloud) | (cloud < 0) | (cloud > 1)
        # 检查源产品本应满足的规则；失败时报错，不静默加入新的删选条件。
        checks = {
            "latitude": np.isfinite(arrays["latitude"]) & (arrays["latitude"] > -60) & (arrays["latitude"] <= 90),
            "longitude": np.isfinite(arrays["longitude"]) & (np.abs(arrays["longitude"]) <= 180),
            "SZA": np.isfinite(arrays["solar_zenith_angle"]) & (arrays["solar_zenith_angle"] < 75),
            "VZA": np.isfinite(arrays["viewing_zenith_angle"]) & (arrays["viewing_zenith_angle"] < 60),
            "chi-square": np.isfinite(arrays["reduced_chi_square"]) & (arrays["reduced_chi_square"] <= 1.2),
            "scan_position": np.isin(arrays["scan_position"], (0, 1, 2)),
        }
        for name, valid in checks.items():
            if np.any(keep & ~valid):
                raise ValueError(f"待保留记录违反源产品规则：{name}")
        expected_corrected = np.asarray(arrays["sif_740"] - arrays["sif_bias"], dtype=np.float32)
        if not np.array_equal(arrays["sif_740_zero_offset_corrected"][keep], expected_corrected[keep], equal_nan=True):
            raise ValueError("源产品校正恒等式不成立")
        attrs = global_metadata(source)
        if target_path.exists():
            output_hash = validate_output(target_path, source, arrays, keep, attrs)
            status = "verified_existing"
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Windows netCDF C库可能不支持中文路径：先在系统临时目录写入，再用Python复制。
            with tempfile.TemporaryDirectory(prefix="gome_lsif_") as temp:
                native_path = Path(temp) / "orbit.nc"
                with nc.Dataset(str(native_path), "w", format="NETCDF4") as out:
                    out.createDimension("record", int(keep.sum()))
                    out.setncatts(attrs)
                    for name, original in source.variables.items():
                        metadata = variable_metadata(name, original)
                        metadata.pop("_FillValue", None)
                        options = {"zlib": True, "complevel": 4} if name != "record" else {}
                        if "_FillValue" in original.ncattrs():
                            options["fill_value"] = original.getncattr("_FillValue")
                        variable = out.createVariable(name, original.dtype, ("record",), **options)
                        variable.set_auto_maskandscale(False)
                        variable.setncatts(metadata)
                        variable[:] = arrays[name][keep]
                output_hash = validate_output(native_path, source, arrays, keep, attrs)
                temporary = target_path.with_name(target_path.name + ".packaging.tmp")
                try:
                    with native_path.open("rb") as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    # 校验最终目录中的副本，成功后再提交为正式 .nc。
                    if validate_output(temporary, source, arrays, keep, attrs) != output_hash:
                        raise ValueError("复制后的文件校验失败")
                    temporary.replace(target_path)
                finally:
                    temporary.unlink(missing_ok=True)
            status = "written"
    after = source_path.stat()
    if (stamp.st_size, stamp.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("读取期间源文件发生变化")
    return dict(file=relative.as_posix(), date=entry["date"], status=status,
                input_records=n, retained_records=int(keep.sum()), removed_records=int((~keep).sum()),
                failed_sif=int((~good_sif).sum()), failed_radiance=int((~good_radiance).sum()),
                failed_both=int((~good_sif & ~good_radiance).sum()),
                invalid_cloud_retained=int((invalid_cloud & keep).sum()),
                source_sha256=source_hash, output_sha256=output_hash, error="")


def user_guide(start_date, end_date, totals):
    return f"""GOME-LSIF v1.0 orbit-level public product user guide

Coverage and organization
-------------------------
The files in this release cover {start_date} to {end_date}.
ERS-2 GOME Band 4 retrievals are stored as:
YYYY/YYYYMMDD/original_GOME_L1b_filename_SIF.nc
Each file contains a compact set of retained land retrievals. Coverage is
spatially and temporally incomplete, particularly after June 2003. Missing
observations are not interpolated or filled with zero. Empty orbit files are
retained and contain zero records; they do not represent zero fluorescence.
Total orbit files: {totals['files']}
Total retained observations: {totals['retained_records']}

Retrieval and correction
------------------------
NADIR observation mode, with east, center and west forward-scan positions.
SIF reference wavelength: 740 nm; retrieval window: 743-758 nm; 72 samples.
Four SVD components and three Legendre polynomial terms are used.
Training spectra use SZA <75 degrees and VZA <60 degrees, with basis functions
constructed separately by processing period and forward-scan position.
Radiometric degradation correction uses the Mali1 quadratic polynomial,
C(t)=P(0)/P(t), referenced to 1995-07-01. The fitted periodic Fourier component
is not applied as part of the radiometric correction factor.
The frozen zero-offset model is:
bias/cos(SZA) = A+B*cos(SZA)+C*mean_radiance+D*cos(latitude)
                +E*cos(SZA)**2+F*mean_radiance**2+G*cos(latitude)**2
Angles supplied in degrees are converted to radians before cosine evaluation.
The estimated normalized bias is multiplied by cos(SZA) to obtain sif_bias.
sif_740_zero_offset_corrected = sif_740 - sif_bias
Publication packaging does not refit either correction model or recalculate SIF.

Retention rules
---------------
The source production product retains NADIR land retrievals with latitude >-60
degrees, SZA <75 degrees, VZA <60 degrees, -5 <= sif_740 <=10, and
reduced_chi_square <=1.2. The publication subset additionally requires:
  -5 <= sif_740 <=5
  25 <= mean_radiance <=200
All four limits above are inclusive. Non-finite values fail these two tests.
The radiance is the degradation-corrected retrieval-window mean.
No additional cloud-fraction screening is applied. Missing or invalid cloud
values from the source product are retained unchanged. No additional range
filter is applied to sif_740_zero_offset_corrected. Scientific applications
may need additional screening appropriate to their purpose.

Variables
---------
All variables have dimension record. The radiometric unit below is
mW m-2 sr-1 nm-1.

record: Original retained-record index in the source SIF product. Values are
        preserved and can be non-consecutive after publication filtering.
scan_index: Zero-based scanline index in the original GOME L1b orbit.
scan_position: 0=east, 1=center/nadir pixel, 2=west; dimensionless.
        NADIR observation mode includes all three positions, not only position 1.
sif_740: SIF at 740 nm after radiometric degradation correction, before
        zero-offset correction; radiometric unit; retained range [-5,5].
sif_standard_error: One-sigma theoretical SIF retrieval uncertainty;
        radiometric unit. Square root of the SIF variance from the least-squares
        covariance matrix; not a variance. Describes retrieval precision, not
        systematic error or correction-model uncertainty. It is not a full
        propagated uncertainty of the final corrected SIF.
sif_bias: Estimated zero-offset bias in the original SIF units; radiometric unit.
sif_740_zero_offset_corrected: Recommended primary SIF variable, after both
        corrections; radiometric unit. Negative values can occur due to noise
        and residual offsets; values are not clipped or converted to zero.
latitude: Footprint-center latitude, degrees_north; -60 < latitude <=90.
longitude: Footprint-center longitude, degrees_east; [-180,180].
mean_radiance: Mean degradation-corrected radiance over the 72 retrieval-window
        samples; radiometric unit; retained range [25,200].
solar_zenith_angle: Solar zenith angle, degrees; <75.
viewing_zenith_angle: Viewing zenith angle, degrees; <60.
solar_azimuth_angle: Source solar azimuth angle, degrees.
viewing_azimuth_angle: Source viewing azimuth angle, degrees.
cloud_fraction: Source GOME footprint cloud fraction; dimensionless. The physical
        fraction domain is [0,1], but missing/invalid source values may remain.
reduced_chi_square: Reduced chi-square of the spectral fit; dimensionless; <=1.2.
        The machine variable name uses ASCII for software compatibility.

Missing values, time and provenance
-----------------------------------
Floating variables retain their source NaN _FillValue. No conversion to -9999
is performed. No per-observation time or footprint-corner variables are included.
Orbit start/end timestamps can be read from the original filename, for example
YYYYMMDDThhmmss_YYYYMMDDThhmmss. They describe the original orbit-file interval,
not exact retained-observation times. Latitude and longitude describe footprint
centers; no footprint boundaries or geographic extent attributes are provided.
NetCDF global attributes are limited to title, product_version, source,
source_file (original L1b basename only), processing_level, observation_date,
spectral_window, production_quality_filter and license. Method details are
provided in this guide; training counts, internal identifiers, hashes and
packaging-history text are not repeated in each NetCDF file.
The publication_manifest.csv lists per-orbit counts, source/output SHA256
checksums and relative orbit paths; run_summary.json records the packaging-script
SHA256 and run times. The original input product retains its processing metadata.
publication_yearly_summary.csv gives annual counts. These are packaging and
consistency checks, not independent scientific validation of the retrieval.
An existing output is reused only after data and metadata validation.
run_summary.json must have status PASS for a complete successful packaging run.

Python example
--------------
import xarray as xr

with xr.open_dataset("path/to/orbit_file.nc") as ds:
    sif = ds["sif_740_zero_offset_corrected"].load()
    latitude = ds["latitude"].load()
    longitude = ds["longitude"].load()
    # Optional application-specific cloud screening (not applied in this release):
    # selected = ds.where(ds["cloud_fraction"] < your_cloud_threshold, drop=True)

Licence
-------
Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/
Institution, contact email and repository identifiers are not supplied here.
"""


def main():
    source_root, output_root = INPUT_ROOT.resolve(), OUTPUT_ROOT.resolve()
    if (source_root == output_root or source_root.is_relative_to(output_root)
            or output_root.is_relative_to(source_root)):
        raise ValueError("输入输出目录必须完全独立，不能相同或互相嵌套")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    entries = inventory(source_root)
    script_hash = sha256(SCRIPT_PATH.read_bytes())
    output_root.mkdir(parents=True, exist_ok=True)
    totals, yearly = Counter(), {}
    summary_path = output_root / "run_summary.json"
    started = utc_now()
    state = dict(status="RUNNING", started=started, expected_files=len(entries),
                 input_root=str(source_root), output_root=str(output_root),
                 packaging_source_sha256=script_hash, retention_filter=FILTER)
    write_json(summary_path, state)
    manifest_tmp = output_root / "publication_manifest.inprogress.csv"
    try:
        with manifest_tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            progress = tqdm(entries, desc="筛选、写出并校验", unit="orbit", dynamic_ncols=True)
            for relative, entry in progress:
                try:
                    row = convert_file(source_root / relative, output_root / relative, relative, entry)
                except Exception as exc:
                    row = dict(file=relative.as_posix(), date=entry["date"], status="ERROR", error=str(exc))
                    totals["errors"] += 1
                    tqdm.write(f"ERROR {relative}: {exc}")
                writer.writerow(row)
                handle.flush()
                if row["status"] != "ERROR":
                    counts = Counter({key: row[key] for key in MANIFEST_FIELDS if isinstance(row.get(key), int)})
                    counts.update(files=1, empty_files=int(row["retained_records"] == 0))
                    totals.update(counts)
                    totals[row["status"]] += 1
                    yearly.setdefault(entry["date"][:4], Counter()).update(counts)
                progress.set_postfix(kept=totals["retained_records"], errors=totals["errors"], refresh=False)
    except BaseException:
        state.update(status="INTERRUPTED", finished=utc_now(), totals=dict(totals))
        write_json(summary_path, state)
        raise
    manifest_tmp.replace(output_root / "publication_manifest.csv")
    state.update(status="FAILED" if totals["errors"] else "PASS", finished=utc_now(), totals=dict(totals))
    state["by_year"] = {year: dict(counts) for year, counts in sorted(yearly.items())}
    # 在指南/汇总也成功写完之后，才将状态提交为 PASS。
    if not totals["errors"]:
        yearly_fields = ["year", "files", "input_records", "retained_records", "removed_records",
                         "failed_sif", "failed_radiance", "failed_both", "invalid_cloud_retained", "empty_files"]
        with (output_root / "publication_yearly_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=yearly_fields, extrasaction="ignore")
            writer.writeheader()
            for year, counts in sorted(yearly.items()):
                writer.writerow(dict(year=year, **counts))
        dates = [entry["date"] for _, entry in entries]
        guide = output_root / "GOME_LSIF_v1_0_user_guide_1995_2011.txt"
        guide.write_text(user_guide(min(dates), max(dates), totals), encoding="utf-8")
    write_json(summary_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    if totals["errors"]:
        raise RuntimeError("部分轨道未通过；详见 publication_manifest.csv。修正问题后重跑，将重新校验已有输出。")
    print(f"完成：{output_root}")


if __name__ == "__main__":
    main()
