# ================== 训练数据构建（仅支持 MODE_NADIR） ==================
import os
import sys
import numpy as np
import netCDF4 as nc
from datetime import datetime
from skimage.transform import resize
from tqdm import tqdm
import gome_sif_utils as f
from global_land_mask import globe

# --- 配置参数 ---
years = list(range(1995, 2012))

start_month = 1
end_month = 12
# 850-734=116，758-734=24
start_wl = 743
end_wl = 758
spectral_num = 72

for year in years:
    print(f"正在处理年份: {year}")
    
    train_folder = fr'F:\traindata_GOME_Band4_NADIR\{year}' 
    os.makedirs(train_folder, exist_ok=True)

    AVHRR_folder = fr'F:\AVHRR_slim\{year}' # 修改：使用精简后的AVHRR数据路径
    if os.path.exists(AVHRR_folder):
        avhrr_files = [
            os.path.join(AVHRR_folder, filename)
            for filename in os.listdir(AVHRR_folder)
            if filename.endswith('.nc')
        ]
    else:
        avhrr_files = []
        print(f"  [警告] 找不到该年份的 AVHRR 数据: {AVHRR_folder}")

    GOME_folder = fr'F:\GOME\{year}'

    # 获取选定的月份文件夹
    all_month_folders = f.get_first_level_folders(GOME_folder)
    selected_month_folders = [mf for mf in all_month_folders if start_month <= int(os.path.basename(mf)) <= end_month]

    all_target_day_folders = []
    for mf in selected_month_folders:
        all_target_day_folders.extend(f.get_first_level_folders(mf))

    for GOME_day_path in tqdm(all_target_day_folders, desc="处理GOME每日数据进度"):
        month_str = os.path.basename(os.path.dirname(GOME_day_path))
        day_str = os.path.basename(GOME_day_path)
        date_str = f"{year}{month_str.zfill(2)}{day_str.zfill(2)}"

        out_file = os.path.join(train_folder, f'{date_str}_traindata.npz')
        if os.path.exists(out_file):
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
                        nadir_files.append(gfile)
            except:
                continue

        if not nadir_files:
            continue

        # 2. 第二步：处理 AVHRR 掩膜
        avhrr_path = next((p for p in avhrr_files if f"{year}{DOY}" in os.path.basename(p)), None)
        if not avhrr_path:
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
        # 裸土 (Soil): NDVI低, 且温度偏高 (> 285K) 意味着地表物理温度必须大于 12℃, 排除了撒哈拉的冷云
        # 加上 Band1 < 0.4 是为了防止和雪（Band1 > 0.4）撞车
        # 基础无植被条件是 NDVI < 0.10 和 Band1 > 0.10，进一步细化为裸土需要满足 BT_11um > 275K 或者 Band1 < 0.4 且 BT_11um > 245K
        # 雪 (Snow): NDVI低, Band1和Band2都较高, 且温度较低 (< 265K) 意味着地表物理温度必须小于 -8℃，排除了撒哈拉的冷云
        # 海洋云 (Ocean Cloud): Band1和Band2都较高, 且温度较低 (< 265K) 意味着地表物理温度必须小于 -8℃，排除了撒哈拉的冷云
        base_soil = (NDVI < 0.10) & (Band1 > 0.10)
        mask_soil = (base_soil & (BT_11um > 275.0)) | (base_soil & (BT_11um > 245.0))
        mask_snow = (NDVI < 0.1) & (Band1 > 0.4) & (Band2 > 0.4) & (BT_11um < 265.0)
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
                    valid = (cf >= 0) & (cf <= 1)
                    cloud_frac_all.append(cf[valid]); lon_all.append(lon[valid]); lat_all.append(lat[valid])
            except: pass

        if not cloud_frac_all: continue
        cloud_fra_grid = f.calculate_grid_mean(np.concatenate(lon_all), np.concatenate(lat_all), np.concatenate(cloud_frac_all), 0.1)

        # 4. 第四步：数据提取主循环
        day_traindata, day_traincols, day_traintype, day_trainlons, day_trainlats, day_trainsza = [], [], [], [], [], []
        
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

                    wl = ds.groups['CALIBRATION'].variables['wavelength'][0, 3]
                    wl_mask = (wl >= start_wl) & (wl <= end_wl)
                    idx = np.where(wl_mask)[0]
                    if len(idx) < spectral_num: continue
                    
                    s_idx = idx[0]
                    e_idx = s_idx + spectral_num
                    rad_sel = rad[:, :, s_idx:e_idx] * 1.986446e-9 / wl[s_idx:e_idx][np.newaxis, np.newaxis, :]

                    n_scan, n_pix = rad_sel.shape[:2]
                    mask = np.zeros((n_scan, n_pix), bool)
                    type_map = -1 * np.ones((n_scan, n_pix), np.int8)

                    for i in range(n_scan):
                        for j in range(n_pix):
                            # 1. 安全检查：如果坐标是 NaN，直接跳过当前像素点
                            if np.isnan(lons[i, j]) or np.isnan(lats[i, j]):
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

                            if t >= 0 and SZA[i, j] < 70 and VZA[i, j] < 60:
                                mask[i, j] = True
                                type_map[i, j] = t

                    if np.any(mask):
                        day_traindata.append(rad_sel[mask])
                        day_traincols.append(np.tile(np.arange(n_pix), (n_scan, 1))[mask])
                        day_traintype.append(type_map[mask])
                        day_trainlons.append(lons[mask])
                        day_trainlats.append(lats[mask])
                        day_trainsza.append(SZA[mask])
            except: pass

        if day_traindata:
            np.savez(out_file,
                     traindata=np.concatenate(day_traindata, axis=0),
                     traincols=np.concatenate(day_traincols, axis=0),
                     traintype=np.concatenate(day_traintype, axis=0),
                     trainlons=np.concatenate(day_trainlons, axis=0), 
                     trainlats=np.concatenate(day_trainlats, axis=0),
                     trainsza=np.concatenate(day_trainsza, axis=0)) 

print("所有年份训练数据完成！勇敢的探索者，数据已就绪。")
