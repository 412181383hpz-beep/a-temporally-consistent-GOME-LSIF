"""Mali1 quadratic training correction; run directly in the gome environment."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


# 在 VSCode 中直接运行；通常只需要修改这一段。
#years = [1995, 1996]
years = list(range(1995, 2012))
start_month = 1
end_month = 12
input_root = Path('outputs/training_raw')
output_root = Path('outputs/training_corrected')
SZA_LIMIT = 75.0
overwrite = False       # 已存在时核对后跳过；True 表示重新生成校正文件。
dry_run = False         # True 只列出待处理文件，不生成结果。


# 原 gome_mali1_quadratic_correction.py 的固定系数与参考日期。
# 只应用长期多项式，不应用拟合时的季节 Fourier 分量。
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


def parse_file_date(path):
    return datetime.strptime(path.name.split('_', 1)[0], '%Y%m%d')


def selected_files(root):
    files = []
    for year in years:
        for path in (root / str(year)).glob('*_traindata.npz'):
            file_date = parse_file_date(path)
            if file_date.year == year and start_month <= file_date.month <= end_month:
                files.append(path)
    return sorted(files)


def validate_source(data, path):
    if set(data) != set(NPZ_KEYS):
        raise ValueError(f'{path}: 必须包含约定的六个变量')
    spectra = data['traindata']
    if spectra.ndim != 2 or spectra.shape[1] != 72:
        raise ValueError(f'{path}: traindata 应为 (N,72)，实际 {spectra.shape}')
    n = len(spectra)
    for key in NPZ_KEYS[1:]:
        if data[key].shape != (n,):
            raise ValueError(f'{path}: {key} 与光谱行数不一致')
    for key in ('traincols', 'traintype'):
        if not np.issubdtype(data[key].dtype, np.integer) or not np.isin(data[key], [0, 1, 2]).all():
            raise ValueError(f'{path}: 无效的 {key}')
    return n


def lineage_path(path):
    return path.with_name(path.name.replace('_traindata.npz', '_lineage.csv'))


def read_lineage(path):
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def filtered_lineage(path, columns, keep):
    rows = read_lineage(path)
    if len(rows) != len(keep):
        raise ValueError(f'{path}: 来源表与 raw NPZ 行数不一致')
    retained = []
    for raw_row, row in enumerate(rows):
        if (int(row['daily_row_index']) != raw_row
                or int(row['detector_column']) != int(columns[raw_row])
                or int(row['scan_index']) < 0 or not row['source_file']):
            raise ValueError(f'{path}: 第 {raw_row} 行来源标识不一致')
        if keep[raw_row]:
            retained.append(dict(row, daily_row_index=str(len(retained)), raw_row_index=str(raw_row)))
    return retained


def correct_one_file(input_path, raw_root, corrected_root, replace_existing=False):
    file_date = parse_file_date(input_path)
    output_path = corrected_root / input_path.relative_to(raw_root)
    source_lineage = lineage_path(input_path)
    output_lineage = lineage_path(output_path)
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    original_count = validate_source(arrays, input_path)
    sza = np.asarray(arrays['trainsza'], dtype=np.float64)
    keep = np.isfinite(sza) & (sza < SZA_LIMIT)
    output_arrays = {key: values[keep] for key, values in arrays.items()}
    spectra = np.asarray(output_arrays['traindata'], dtype=np.float64).copy()
    columns = output_arrays['traincols']
    factors = correction_factors(file_date)
    for fov_index, factor in enumerate(factors):
        spectra[columns == fov_index] *= factor
    output_arrays['traindata'] = spectra
    for key, values in output_arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f'{input_path}: 校正后 {key} 含非有限值')
    lineage_rows = filtered_lineage(source_lineage, arrays['traincols'], keep)

    # 即使文件已存在，也核对当前 raw、系数和来源，不直接跳过。
    exists = output_path.exists() or output_lineage.exists()
    if exists and not replace_existing:
        if not output_path.exists() or not output_lineage.exists():
            raise ValueError(f'{output_path}: 已有结果不完整，请检查后决定是否 overwrite')
        with np.load(output_path, allow_pickle=False) as saved:
            if set(saved.files) != set(NPZ_KEYS):
                raise ValueError(f'{output_path}: 已有结果变量不一致')
            for key, expected in output_arrays.items():
                if saved[key].dtype != expected.dtype or not np.array_equal(saved[key], expected):
                    raise ValueError(f'{output_path}: 已有 {key} 与当前输入/系数不一致')
        if read_lineage(output_lineage) != lineage_rows:
            raise ValueError(f'{output_lineage}: 已有来源映射不一致')
        status = 'verified_existing'
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + '.tmp.npz')
        temporary_lineage = output_lineage.with_name(output_lineage.name + '.tmp')
        try:
            np.savez(temporary_path, **output_arrays)
            with temporary_lineage.open('w', newline='', encoding='utf-8-sig') as stream:
                fields = ['daily_row_index', 'source_file', 'scan_index', 'detector_column', 'raw_row_index']
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(lineage_rows)
            temporary_lineage.replace(output_lineage)
            temporary_path.replace(output_path)
        finally:
            for temporary in (temporary_path, temporary_lineage):
                if temporary.exists():
                    temporary.unlink()
        status = 'written' if keep.any() else 'written_empty_sza_filter'

    row = {
        'date': file_date.strftime('%Y-%m-%d'), 'input_file': str(input_path),
        'output_file': str(output_path), 'status': status,
        'original_records': original_count, 'retained_sza_lt_limit': int(keep.sum()),
        'removed_sza_ge_limit_or_invalid': int((~keep).sum()), 'sza_limit': SZA_LIMIT,
        'sza_70_75': int(np.sum((sza >= 70) & keep)),
        'reference_date': REFERENCE_DATE.strftime('%Y-%m-%d'),
    }
    for index, name in enumerate(FOV_NAMES):
        row[name + '_records'] = int(np.sum(columns == index))
        row[name + '_correction_factor'] = float(factors[index])
    return row


def write_manifest(rows):
    if not rows:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    label = '_'.join(str(year) for year in years)
    path = output_root / f'traindata_manifest_{label}_months_{start_month:02d}_{end_month:02d}.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw, corrected = input_root.resolve(), output_root.resolve()
    if raw == corrected or corrected.is_relative_to(raw) or raw.is_relative_to(corrected):
        raise ValueError('raw 和 corrected 必须保存在独立目录')
    if not 1 <= start_month <= end_month <= 12:
        raise ValueError('月份范围应满足 1 <= start_month <= end_month <= 12')
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    files = selected_files(input_root)
    print(f'共找到 {len(files)} 个训练文件；年份={years}，月份={start_month}-{end_month}')
    if not files:
        raise FileNotFoundError('未找到指定年月的训练 NPZ；请先运行训练数据脚本')
    rows = []
    for index, path in enumerate(files, start=1):
        if dry_run:
            print(f'[DRY RUN] {path} -> {output_root / path.relative_to(input_root)}')
            continue
        try:
            row = correct_one_file(path, input_root, output_root, overwrite)
        except Exception as exc:
            rows.append({'input_file': str(path), 'status': 'failed', 'error': str(exc)})
            write_manifest(rows)
            raise
        rows.append(row)
        write_manifest(rows)
        print(f'[{index}/{len(files)}] {row["status"]}: {path.name}; '
              f'N={row["retained_sza_lt_limit"]}, 70<=SZA<75={row["sza_70_75"]}')
    print('预览结束，没有写入数据。' if dry_run else '训练光谱二阶校正完成，raw 文件保持不变。')


if __name__ == '__main__':
    main()
