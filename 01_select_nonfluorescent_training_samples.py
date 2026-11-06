# ================== 训练数据构建（仅支持 MODE_NADIR） ==================
import os
import csv
import traceback
import numpy as np
import netCDF4 as nc
from datetime import datetime
from skimage.transform import resize
from tqdm import tqdm
import gome_sif_utils as f
from global_land_mask import globe

def to_nan_array(x):
    array = np.asanyarray(x)
    if np.ma.isMaskedArray(array):
        if not np.issubdtype(array.dtype, np.inexact):
            array = array.astype(float)
        return array.filled(np.nan)
    return np.asarray(array)


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

# --- 配置参数 ---
#years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003]  # 示例年份，替换为实际需要的年份
years = list(range(1995, 2012))
#years = [1995,1996]

start_month = 1
end_month = 12
SZA_MAX = 75.0
VZA_MAX = 60.0
train_root = 'outputs/training_raw'
GOME_ROOT = 'data/GOME'
AVHRR_ROOT = 'data/AVHRR'
# 850-734=116，758-734=24
start_wl = 743
end_wl = 758
spectral_num = 72

for year in years:
    print(f"正在处理年份: {year}")
    
    train_folder = os.path.join(train_root, str(year)) 
    os.makedirs(train_folder, exist_ok=True)

    AVHRR_folder = os.path.join(AVHRR_ROOT, str(year)) # 修改：使用精简后的AVHRR数据路径
    if os.path.exists(AVHRR_folder):
        avhrr_files = sorted(os.path.join(AVHRR_folder, name) for name in os.listdir(AVHRR_folder) if name.endswith('.nc'))
    else:
        avhrr_files = []
        print(f"  [警告] 找不到该年份的 AVHRR 数据: {AVHRR_folder}")

    GOME_folder = os.path.join(GOME_ROOT, str(year))

    # 获取选定的月份文件夹
    all_month_folders = f.get_first_level_folders(GOME_folder)
    # 首尾年份限定为 1995 年 6 月至 2011 年 6 月。
    first_month = 6 if year == 1995 else start_month
    last_month = 6 if year == 2011 else end_month
    selected_month_folders = [mf for mf in all_month_folders if first_month <= int(os.path.basename(mf)) <= last_month]

    all_target_day_folders = []
    for mf in selected_month_folders:
        all_target_day_folders.extend(f.get_first_level_folders(mf))

    for GOME_day_path in tqdm(all_target_day_folders, desc="处理GOME每日数据进度"):
        month_str = os.path.basename(os.path.dirname(GOME_day_path))
        day_str = os.path.basename(GOME_day_path)
        date_str = f"{year}{month_str.zfill(2)}{day_str.zfill(2)}"

        out_file = os.path.join(train_folder, f'{date_str}_traindata.npz')
        lineage_file = out_file.replace('_traindata.npz', '_lineage.csv')
        if os.path.exists(out_file):
            if not os.path.exists(lineage_file):
                raise RuntimeError(f'已有训练文件缺少来源表，请先检查：{out_file}')
            print(f'[SKIP] 已存在: {out_file}')
            continue

        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            DOY = str(date_obj.timetuple().tm_yday).zfill(3)
        except:
            continue

        # 1. 第一步：预筛选当天的 NADIR 文件
        nadir_files = []
        all_nc_files = f.get_nc_files(GOME_day_path)
        for gfile in all_nc_files:
            try:
                with nc.Dataset(gfile) as ds:
                    if 'MODE_NADIR' in ds.groups and 'BAND_4' in ds.groups['MODE_NADIR'].groups:
                        geometry_ok, shapes = nadir_geometry(ds.groups['MODE_NADIR'].groups['BAND_4'])
                        if not geometry_ok:
                            print(f'[SKIP_NONSTANDARD_GEOMETRY] {gfile} | shapes={shapes}')
                            geometry_qa = os.path.join(train_folder, 'training_orbit_QA.csv')
                            new_qa = not os.path.exists(geometry_qa)
                            with open(geometry_qa, 'a', newline='', encoding='utf-8-sig') as stream:
                                writer = csv.writer(stream)
                                if new_qa:
                                    writer.writerow(['date', 'source_file', 'status', 'shapes'])
                                writer.writerow([date_str, gfile, 'skipped_nonstandard_geometry', str(shapes)])
                            continue
                        nadir_files.append(gfile)
            except Exception as e:
                print(f"[ERROR] NADIR read failed: {gfile} | {type(e).__name__}: {e}")
                traceback.print_exc()
                raise

        if not nadir_files:
            print(f'[NO_NADIR] {date_str}: 没有 NADIR/BAND_4 数据')
            continue

        # 2. 第二步：处理 AVHRR 掩膜
        avhrr_path = next((p for p in avhrr_files if f"{year}{DOY}" in os.path.basename(p)), None)
        if not avhrr_path:
            print(f'[SKIP_MISSING_AVHRR] {date_str}: 找不到同日 AVHRR 文件，跳过当天训练数据生成')
            qa_file = os.path.join(train_folder, 'training_daily_QA.csv')
            new_qa = not os.path.exists(qa_file)
            with open(qa_file, 'a', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                if new_qa:
                    writer.writerow(['date', 'status', 'output_file', 'avhrr_file',
                                     'nadir_orbits', 'zero_sample_orbits', 'N', 'Soil_N',
                                     'Cloud_N', 'Snow_N', 'SZA_LT70_N', 'SZA_70_75_N', 'max_SZA'])
                writer.writerow([date_str, 'SKIP_MISSING_AVHRR', '', '', len(nadir_files),
                                 '', '', '', '', '', '', '', ''])
            continue

        ds_avhrr = nc.Dataset(avhrr_path)
        def process_avhrr_band(band_data):
            return band_data.filled(np.nan).astype(np.float32) if np.ma.isMaskedArray(band_data) else np.where(band_data <= -9000, np.nan, band_data).astype(np.float32)

        Band1 = resize(process_avhrr_band(ds_avhrr.variables['SREFL_CH1'][:]), (1800, 3600), order=1, preserve_range=True)
        Band2 = resize(process_avhrr_band(ds_avhrr.variables['SREFL_CH2'][:]), (1800, 3600), order=1, preserve_range=True)
        BT_11um = resize(process_avhrr_band(ds_avhrr.variables['BT_CH4'][:]), (1800, 3600), order=1, preserve_range=True)
        ds_avhrr.close()

        with np.errstate(divide='ignore', invalid='ignore'):
            NDVI = (Band2 - Band1) / (Band2 + Band1)

        # 掩膜定义
        # Soil 温度条件保留原写法，实际等价于 BT_11um > 245 K。
        # 类别编码：0=Soil、1=Cloud、2=Snow；重叠时优先 Soil。
        base_soil = (NDVI < 0.15) & (Band1 > 0.10)
        mask_soil = (base_soil & (BT_11um > 275.0)) | (base_soil & (BT_11um > 245.0))
        mask_snow = (NDVI < 0.15) & (Band1 > 0.4) & (Band2 > 0.4) & (BT_11um < 265.0)
        mask_ocean_cloud = (Band1 > 0.6) & (Band2 > 0.6) & (BT_11um < 265.0)

        # 3. 第三步：GOME 云量辅助（仅针对筛选出的 nadir_files）
        cloud_frac_all, lon_all, lat_all = [], [], []
        for gfile in nadir_files:
            try:
                with nc.Dataset(gfile) as ds:
                    mode = ds.groups['MODE_NADIR']
                    cf = mode.groups['BAND_4'].groups['CLOUDDATA'].variables['cloud_fraction'][0]
                    lon = mode.groups['BAND_4'].groups['GEODATA'].variables['longitude'][0]
                    lat = mode.groups['BAND_4'].groups['GEODATA'].variables['latitude'][0]
                    cf, lon, lat = to_nan_array(cf), to_nan_array(lon), to_nan_array(lat)
                    valid = np.isfinite(cf) & np.isfinite(lon) & np.isfinite(lat) & (cf >= 0) & (cf <= 1)
                    cloud_frac_all.append(cf[valid]); lon_all.append(lon[valid]); lat_all.append(lat[valid])
            except Exception as e:
                print(f"[ERROR] Cloud read failed: {gfile} | {type(e).__name__}: {e}")
                traceback.print_exc()
                raise

        if not cloud_frac_all: continue
        cloud_fra_grid = f.calculate_grid_mean(np.concatenate(lon_all), np.concatenate(lat_all), np.concatenate(cloud_frac_all), 0.1)

        # 4. 第四步：数据提取主循环
        day_traindata, day_traincols, day_traintype, day_trainlons, day_trainlats, day_trainsza = [], [], [], [], [], []
        day_lineage = []
        zero_sample_orbits = 0
        
        for gfile in nadir_files:
            try:
                with nc.Dataset(gfile) as ds:
                    mode = ds.groups['MODE_NADIR']
                    # NADIR 模式固定窗口大小
                    lon_offset, lat_offset = 18, 3
                    
                    band4 = mode.groups['BAND_4']
                    rad = band4.groups['OBSERVATIONS'].variables['radiance'][0]
                    lons = band4.groups['GEODATA'].variables['longitude'][0]
                    lats = band4.groups['GEODATA'].variables['latitude'][0]
                    SZA = band4.groups['GEODATA'].variables['solar_zenith_angle'][0, :, :, 1]
                    VZA = band4.groups['GEODATA'].variables['viewing_zenith_angle'][0, :, :, 1]

                    rad = to_nan_array(rad)
                    lons = to_nan_array(lons)
                    lats = to_nan_array(lats)
                    SZA = to_nan_array(SZA)
                    VZA = to_nan_array(VZA)

                    wl = to_nan_array(ds.groups['CALIBRATION'].variables['wavelength'][0, 3])
                    wl_mask = (wl >= start_wl) & (wl <= end_wl)
                    idx = np.where(wl_mask)[0]
                    if len(idx) < spectral_num:
                        raise ValueError(f'{gfile}: 743-758 nm 通道少于 {spectral_num}')
                    
                    s_idx = idx[0]
                    e_idx = s_idx + spectral_num
                    rad_sel = rad[:, :, s_idx:e_idx] * 1.986446e-9 / wl[s_idx:e_idx][np.newaxis, np.newaxis, :]

                    rad_sel = to_nan_array(rad_sel)
                    n_scan, n_pix = rad_sel.shape[:2]
                    mask = np.zeros((n_scan, n_pix), bool)
                    type_map = -1 * np.ones((n_scan, n_pix), np.int8)

                    for i in range(n_scan):
                        for j in range(n_pix):
                            # Skip only the observation with invalid coordinates, geometry or spectrum.
                            if not (np.isfinite(lons[i, j]) and np.isfinite(lats[i, j])):
                                continue
                            if not (np.isfinite(SZA[i, j]) and np.isfinite(VZA[i, j])):
                                continue
                            if not np.all(np.isfinite(rad_sel[i, j, :])):
                                continue
                            # 2. 修正索引计算
                            # 经度：-180 -> 0, 180 -> 3600
                            ilon = int(np.floor((lons[i, j] + 180) / 0.1))
                            # 纬度：注意 flipud 之后，90° 对应索引 0，-90° 对应 1800
                            ilat = int(np.floor((90 - lats[i, j]) / 0.1))

                            # 3. 边界保护
                            ilon = max(0, min(3599, ilon))
                            ilat = max(0, min(1799, ilat))

                            lat_s, lat_e = max(0, ilat - lat_offset), min(1800, ilat + lat_offset + 1)
                            lon_s, lon_e = max(0, ilon - lon_offset), min(3600, ilon + lon_offset + 1)
                            # 提取各个掩码在当前经纬度窗口内的占比
                            soil_ratio = np.mean(mask_soil[lat_s:lat_e, lon_s:lon_e])
                            snow_ratio = np.mean(mask_snow[lat_s:lat_e, lon_s:lon_e])
                            ocean_cloud_ratio = np.mean(mask_ocean_cloud[lat_s:lat_e, lon_s:lon_e])
                            sub_c = cloud_fra_grid[lat_s:lat_e, lon_s:lon_e]
                            # c_m = cloud fraction 的局部平均值，用于辅助判断云量情况，尤其是对于海洋云和裸土的区分
                            # 只有当 size > 0 且 并不全是 NaN 时，才进行计算
                            if sub_c.size > 0 and not np.all(np.isnan(sub_c)):
                                cloud_mean = np.nanmean(sub_c)
                            else:
                                cloud_mean = np.nan

                            t = -1
                            # === 海陆分流逻辑 ===
                            if globe.is_land(lats[i, j], lons[i, j]):
                                # 【陆地上】只提取 裸土(0) 和 雪(2)
                                # 无论是看土还是看雪，天空都必须是相对晴朗的 (cloud_mean < 0.3)
                                # s_r soil ratio, sn_r snow ratio, oc_r ocean cloud ratio
                                if cloud_mean < 0.3:
                                    if soil_ratio > 0.7: t = 0
                                    elif snow_ratio > 0.7: t = 2
                            else:
                                if ocean_cloud_ratio > 0.7 and cloud_mean > 0.95: t = 1

                            if t >= 0 and SZA[i, j] < SZA_MAX and VZA[i, j] < VZA_MAX:
                                mask[i, j] = True
                                type_map[i, j] = t

                    if np.any(mask):
                        day_traindata.append(rad_sel[mask].astype(np.float32))
                        day_traincols.append(np.tile(np.arange(n_pix), (n_scan, 1))[mask])
                        day_traintype.append(type_map[mask])
                        day_trainlons.append(lons[mask].astype(np.float32))
                        day_trainlats.append(lats[mask].astype(np.float32))
                        day_trainsza.append(SZA[mask].astype(np.float32))
                        scans, columns = np.where(mask)
                        day_lineage.extend((gfile, int(scan), int(col)) for scan, col in zip(scans, columns))
                    else:
                        zero_sample_orbits += 1
            except Exception as e:
                print(f"[ERROR] Failed orbit: {gfile} | {type(e).__name__}: {e}")
                traceback.print_exc()
                raise

        if day_traindata:
            arrays = dict(
                traindata=np.concatenate(day_traindata, axis=0),
                traincols=np.concatenate(day_traincols, axis=0),
                traintype=np.concatenate(day_traintype, axis=0),
                trainlons=np.concatenate(day_trainlons, axis=0),
                trainlats=np.concatenate(day_trainlats, axis=0),
                trainsza=np.concatenate(day_trainsza, axis=0))
            n = len(arrays['traindata'])
            if arrays['traindata'].shape != (n, spectral_num) or len(day_lineage) != n:
                raise ValueError(f'{date_str}: 光谱或来源表行数不一致')
            tmp_file = out_file + '.tmp.npz'
            tmp_lineage = lineage_file + '.tmp'
            np.savez(tmp_file, **arrays)
            with open(tmp_lineage, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow(['daily_row_index', 'source_file', 'scan_index', 'detector_column'])
                writer.writerows((row, *source) for row, source in enumerate(day_lineage))
            os.replace(tmp_lineage, lineage_file)
            os.replace(tmp_file, out_file)
            types = arrays['traintype']
            sza = arrays['trainsza']
            counts = [int(np.sum(types == code)) for code in (0, 1, 2)]
            print(f'[WRITTEN] {date_str}: N={n}, Soil/Cloud/Snow={counts}, '
                  f'70<=SZA<75={np.sum(sza >= 70)}, max SZA={np.max(sza):.4f}')
        else:
            n, counts = 0, [0, 0, 0]
            sza = np.array([], dtype=np.float32)
            print(f'[NO_SAMPLES] {date_str}: 筛选后没有样本，不写空 NPZ')

        # 简单日清单放在年份目录；来源信息单独保存，不增加训练 NPZ 变量。
        qa_file = os.path.join(train_folder, 'training_daily_QA.csv')
        new_qa = not os.path.exists(qa_file)
        with open(qa_file, 'a', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            if new_qa:
                writer.writerow(['date', 'status', 'output_file', 'avhrr_file',
                                 'nadir_orbits', 'zero_sample_orbits', 'N', 'Soil_N',
                                 'Cloud_N', 'Snow_N', 'SZA_LT70_N', 'SZA_70_75_N', 'max_SZA'])
            writer.writerow([date_str, 'WRITTEN' if n else 'NO_SAMPLES',
                             out_file if n else '', avhrr_path, len(nadir_files),
                             zero_sample_orbits, n, *counts, int(np.sum(sza < 70)),
                             int(np.sum(sza >= 70)), float(np.max(sza)) if n else ''])

print("所选年份和月份处理结束；请查看 WRITTEN / NO_NADIR / NO_SAMPLES / SKIP_MISSING_AVHRR 日志。")
