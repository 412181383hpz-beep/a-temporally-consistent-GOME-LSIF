import os
import sys
import numpy as np
import netCDF4 as nc
from tqdm import tqdm
from datetime import datetime
from scipy.special import eval_legendre
from global_land_mask import globe
import gome_sif_utils as f

# =================================================================
# 1. 全局配置与参数设置 (Phase 2: 2003-2011 衰退求生期)
# =================================================================
YEARS = list(range(2003, 2012))
MONTHS = list(range(1, 13))

# 🌟 核心时间切分与滑动窗口参数 🌟
TRAIN_WINDOW = 15     # 目标滑动窗口大小：15 天
MIN_TAIL_DAYS = 5     # 尾部独立成组最低门槛
MELTDOWN_DAYS = 7     # 熔断底线：全月少于7天直接放弃

# 🌟 核心抽样引擎参数 🌟
LAT_STEP = 10         
SOIL_BIN_CAP = 160    
CLOUD_SOIL_RATIO = 1.0 
MIN_SAFE_TOTAL = 200  # ★ SVD 满秩保命底线 ★

# 物理与光谱参数
start_wl = 743
end_wl = 758
npc = 4                
npp = 3                
num_spectral = 72      
num_column = 3         

# 路径设置
TRAIN_ROOT = r'F:\traindata_GOME_Band4_corrected_fourthorder'
GOME_ROOT = r'F:\GOME'
SAVE_ROOT = r'F:\SIF740_GOME_N4_corrected_FourthOrder'

# =================================================================
# 2. 四阶衰减校正 (Dfactor) 模块
# =================================================================
PIXEL_COEFFS = {
    0: [-11.248, 173.150, -998.196, 2554.176, -2446.991], 
    1: [-12.326, 186.948, -1062.173, 2679.343, -2531.249], 
    2: [-2.160, 32.928, -188.111, 477.258, -453.152]   
}

def get_dfactor(current_date, pixel_type):
    coeffs = PIXEL_COEFFS.get(pixel_type, PIXEL_COEFFS[1])
    poly_func = np.poly1d(coeffs)
    
    ref_date_1900 = datetime(1900, 1, 1)
    baseline_date = datetime(1995, 7, 1)
    
    baseline_nod = (baseline_date - ref_date_1900).days
    current_nod = (current_date - ref_date_1900).days
    
    # 【核心修复】统一为 10000.0
    x_current = current_nod / 10000.0
    x_baseline = baseline_nod / 10000.0
    
    return poly_func(x_current) / poly_func(x_baseline)

def get_active_mode_group(dataset):
    if 'MODE_NADIR' in dataset.groups:
        mode_grp = dataset.groups['MODE_NADIR']
        if 'BAND_4' in mode_grp.groups:
            return mode_grp
    return None

# =================================================================
# 3. 超长周期主循环 (滑动解耦窗口机制)
# =================================================================
for year in YEARS:
    # 2003年从7月开始跑，避免和 Phase 1 重复
    run_months = range(7, 13) if year == 2003 else MONTHS
    for month in run_months:
        traindata_folder = os.path.join(TRAIN_ROOT, str(year))
        GOME_month_folder = os.path.join(GOME_ROOT, str(year), str(month).zfill(2))
        save_folder = os.path.join(SAVE_ROOT, str(year))
        
        if not os.path.exists(GOME_month_folder):
            continue
            
        os.makedirs(save_folder, exist_ok=True)

        all_days_paths = f.get_first_level_folders(GOME_month_folder)
        def get_date_from_path(p):
            return datetime(year, int(p.split(os.sep)[-2]), int(os.path.basename(p)))
        all_days_paths.sort(key=get_date_from_path)
        
        valid_days_paths = []
        for p in all_days_paths:
            d_str = os.path.basename(p)
            t_path = os.path.join(traindata_folder, f'{year}{str(month).zfill(2)}{d_str.zfill(2)}_traindata.npz')
            if os.path.exists(t_path):
                valid_days_paths.append(p)
                
        # 熔断机制：全月有效数据少于底线
        if len(valid_days_paths) < MELTDOWN_DAYS:
            if len(valid_days_paths) > 0:
                print(f"⚠️ {year}-{str(month).zfill(2)} 全月仅存 {len(valid_days_paths)} 天数据，触发熔断跳过。")
            continue

        print(f"\n=== 🚀 开始反演 (Phase 2): {year} 年 {month} 月 (含 {len(valid_days_paths)} 个有效日) ===")
        
        # --- 第一步：划分反演组 (15天一组) ---
        inv_groups = []
        curr_group = []
        for p in valid_days_paths:
            curr_group.append(p)
            if len(curr_group) == TRAIN_WINDOW:
                inv_groups.append(curr_group)
                curr_group = []
                
        if curr_group:
            if len(inv_groups) == 0:
                inv_groups.append(curr_group)
            elif len(curr_group) >= MIN_TAIL_DAYS:
                inv_groups.append(curr_group)
            else:
                inv_groups[-1].extend(curr_group)

        # --- 第二步：为每个反演组匹配对应的滑动训练组 (向前借用) ---
        tasks = []
        for inv_g in inv_groups:
            if len(inv_g) >= TRAIN_WINDOW:
                train_g = inv_g
            else:
                last_day_path = inv_g[-1]
                last_idx = valid_days_paths.index(last_day_path)
                start_idx = max(0, last_idx - TRAIN_WINDOW + 1)
                train_g = valid_days_paths[start_idx : last_idx + 1]
            
            tasks.append({'train_files': train_g, 'inv_files': inv_g})

        # =================================================================
        # 4. 核心周期级反演处理
        # =================================================================
        for task in tqdm(tasks, desc=f"【{year}-{str(month).zfill(2)}】周期进度"):
            train_files = task['train_files']
            inv_files = task['inv_files']
            
            dates = [os.path.basename(folder) for folder in inv_files]
            date_str = f"{year}{str(month).zfill(2)}{dates[0].zfill(2)}"
            basis_save_path = os.path.join(save_folder, f'SVD_Basis_{date_str}.npz')
            
            all_days_finished = True
            for day_folder in inv_files:
                d_str = os.path.basename(day_folder)
                cur_date = f"{year}{str(month).zfill(2)}{d_str.zfill(2)}"
                if not os.path.exists(os.path.join(save_folder, f"SIF_GOME_{cur_date}_Daily.nc")):
                    all_days_finished = False
                    break
            
            if os.path.exists(basis_save_path) and all_days_finished:
                continue

            # --- SVD 训练 (基于向前借用的 15 天滑动窗) ---
            if os.path.exists(basis_save_path):
                tmp_svd = np.load(basis_save_path)
                V_all, S_all = tmp_svd['V_all'], tmp_svd['S_all']
            else:
                V_all = np.full((num_column, num_spectral, num_spectral), np.nan)
                S_all = np.full((num_column, num_spectral), np.nan)
                
                train_list, lats_list, cols_list, types_list = [], [], [], [] 
                for day_folder in train_files:
                    d = os.path.basename(day_folder)
                    t_path = os.path.join(traindata_folder, f'{year}{str(month).zfill(2)}{d.zfill(2)}_traindata.npz')
                    tmp = np.load(t_path)
                    if all(k in tmp for k in ['traindata', 'trainlats', 'traincols', 'traintype']):
                        train_list.append(tmp['traindata'])
                        lats_list.append(tmp['trainlats'])
                        cols_list.append(tmp['traincols'])
                        types_list.append(tmp['traintype'])

                if train_list:
                    raw_train = np.concatenate(train_list)
                    raw_lats = np.concatenate(lats_list)
                    raw_cols = np.concatenate(cols_list)
                    raw_types = np.concatenate(types_list)
                    
                    for col_idx in range(num_column):
                        c_mask = (raw_cols == col_idx)
                        c_train, c_lats, c_types = raw_train[c_mask], raw_lats[c_mask], raw_types[c_mask]
                        if len(c_train) < 100: continue

                        lat_bins = np.arange(-90, 91, LAT_STEP)
                        bins_data = []
                        for i in range(len(lat_bins)-1):
                            bin_mask = (c_lats >= lat_bins[i]) & (c_lats < lat_bins[i+1])
                            bins_data.append({
                                'soil_idx': np.where((c_types == 0) & bin_mask)[0],
                                'cloud_idx': np.where((c_types == 1) & bin_mask)[0],
                                'snow_idx': np.where((c_types == 2) & bin_mask)[0],
                            })

                        selected_data_idx = []
                        total_sampled_soil = 0
                        
                        for b in bins_data:
                            if len(b['soil_idx']) > SOIL_BIN_CAP:
                                samp_soil = np.random.choice(b['soil_idx'], SOIL_BIN_CAP, replace=False)
                            else:
                                samp_soil = b['soil_idx']
                            selected_data_idx.extend(samp_soil)
                            total_sampled_soil += len(samp_soil)
                            selected_data_idx.extend(b['snow_idx'])

                        # 🌟 极限保命机制介入 🌟
                        target_cloud_total = int(total_sampled_soil * CLOUD_SOIL_RATIO)
                        total_sampled_snow = sum([len(b['snow_idx']) for b in bins_data])
                        
                        current_planned_total = total_sampled_soil + total_sampled_snow + target_cloud_total
                        if current_planned_total < MIN_SAFE_TOTAL:
                            # 数据荒：突破 1:1 限制，强制索要云端名额保底
                            target_cloud_total = MIN_SAFE_TOTAL - total_sampled_soil - total_sampled_snow
                            total_available_cloud = sum([len(b['cloud_idx']) for b in bins_data])
                            target_cloud_total = min(target_cloud_total, total_available_cloud)
                            
                        remaining_cloud_quota = target_cloud_total
                        
                        sorted_bins = sorted(bins_data, key=lambda x: len(x['cloud_idx']))
                        for i, b in enumerate(sorted_bins):
                            bins_left = len(sorted_bins) - i
                            if bins_left == 0: break
                            current_quota = remaining_cloud_quota // bins_left
                            take = min(len(b['cloud_idx']), current_quota)
                            if take > 0:
                                samp_cloud = np.random.choice(b['cloud_idx'], take, replace=False)
                                selected_data_idx.extend(samp_cloud)
                            remaining_cloud_quota -= take

                        if len(selected_data_idx) > 0:
                            final_train = c_train[selected_data_idx]
                            if len(final_train) >= 100:
                                final_train = f.filter_outliers_with_pca(final_train)
                                _, S, V = np.linalg.svd(final_train, full_matrices=False)
                                V_all[col_idx], S_all[col_idx] = V, S

                np.savez(basis_save_path, V_all=V_all, S_all=S_all)

            # --- SIF 反演 (严格仅对本周期 inv_files 进行反演) ---
            for day_path in inv_files:
                d_str = os.path.basename(day_path)
                cur_date = f"{year}{str(month).zfill(2)}{d_str.zfill(2)}"
                current_date_obj = datetime(year, month, int(d_str))
                
                day_nc = os.path.join(save_folder, f"SIF_GOME_{cur_date}_Daily.nc")
                if os.path.exists(day_nc): continue
                    
                save_date_folder = os.path.join(save_folder, cur_date)
                os.makedirs(save_date_folder, exist_ok=True)
                
                gome_files = f.get_nc_files(day_path)
                if not gome_files: continue
                
                cloud_frac_all, lon_all, lat_all = [], [], []
                for gfile in gome_files:
                    try:
                        with nc.Dataset(gfile) as ds:
                            mode_grp = get_active_mode_group(ds)
                            if not mode_grp: continue
                            cf = mode_grp.groups['BAND_4'].groups['CLOUDDATA'].variables['cloud_fraction'][0]
                            ln = mode_grp.groups['BAND_4'].groups['GEODATA'].variables['longitude'][0]
                            lt = mode_grp.groups['BAND_4'].groups['GEODATA'].variables['latitude'][0]
                            v = (cf >= 0) & (cf <= 1)
                            cloud_frac_all.append(cf[v]); lon_all.append(ln[v]); lat_all.append(lt[v])
                    except: pass

                if not cloud_frac_all: continue
                cloud_fra_grid = f.calculate_grid_mean(np.concatenate(lon_all), np.concatenate(lat_all), np.concatenate(cloud_frac_all), 0.1)
                lon_sequence, lat_sequence = f.lon_lat_arange(0.1)
                
                for gfile in gome_files:
                    fname = os.path.splitext(os.path.basename(gfile))[0]
                    out_nc = os.path.join(save_date_folder, f"{fname}_SIF.nc")
                    if os.path.exists(out_nc): continue

                    with nc.Dataset(gfile) as ds:
                        m_grp = get_active_mode_group(ds)
                        if not m_grp: continue
                        b4, g4, w4 = m_grp.groups['BAND_4'], m_grp.groups['BAND_4'].groups['GEODATA'], ds.groups['CALIBRATION']
                        
                        rad = b4.groups['OBSERVATIONS'].variables['radiance'][0]
                        lons, lats = g4.variables['longitude'][0], g4.variables['latitude'][0]
                        SZA, VZA = g4.variables['solar_zenith_angle'][0,:,:,1], g4.variables['viewing_zenith_angle'][0,:,:,1]
                        SAA, VAA = g4.variables['solar_azimuth_angle'][0,:,:,1], g4.variables['viewing_azimuth_angle'][0,:,:,1]
                        t_arr = b4.groups['OBSERVATIONS'].variables['delta_time'][0, :] / 1000 + b4.groups['OBSERVATIONS'].variables['time'][0]
                        
                        wl = w4.variables['wavelength'][0, 3, :]
                        idx_s = np.where((wl >= start_wl) & (wl <= end_wl))[0][0]
                        wl_crop = wl[idx_s : idx_s + num_spectral]
                        rad_crop = rad[:, :, idx_s : idx_s + num_spectral]
                        
                        curr_num_cols = lons.shape[1]
                        dfactor_array = np.array([get_dfactor(current_date_obj, p) for p in range(curr_num_cols)])
                        rad_e = (rad_crop * 1.986446e-9 / wl_crop[np.newaxis, np.newaxis, :]) / dfactor_array[np.newaxis, :, np.newaxis]
                        
                        poly_all = np.zeros((num_column, num_spectral, npc + npp + 1))
                        wn = 2 * (wl_crop - np.min(wl_crop)) / (np.max(wl_crop) - np.min(wl_crop)) - 1
                        hf = np.exp(-(wl_crop - 740)**2 / (2 * 10**2))
                        for c_i in range(num_column):
                            if np.isnan(V_all[c_i, 0, 0]): continue
                            pm = np.zeros((num_spectral, npc + npp))
                            for p in range(npc): pm[:, p] = V_all[c_i][p, :]
                            for p in range(npp): pm[:, p + npc] = eval_legendre(p + 1, wn) * V_all[c_i][0, :]
                            poly_all[c_i] = np.hstack((pm, hf[:, np.newaxis]))
                        
                        sif, err, ka2 = np.full(lons.shape, np.nan, "f4"), np.full(lons.shape, np.nan, "f4"), np.full(lons.shape, np.nan, "f4")
                        rad_m, time_s = np.full(lons.shape, np.nan, "f4"), np.full(lons.shape, np.nan, "f4")
                        
                        for j in range(curr_num_cols):
                            if j >= num_column or np.all(poly_all[j] == 0): continue
                            for i in range(lons.shape[0]):
                                # 【南极洲剔除】
                                if globe.is_land(lats[i,j], lons[i,j]) and lats[i,j] > -60 and SZA[i,j] < 70 and VZA[i,j] < 60:
                                    try:
                                        s_v, chi2_v, e_v = f.sif_retrieval(poly_all[j], rad_e[i,j,:], npc+npp+1)
                                        sif[i,j], ka2[i,j], err[i,j], rad_m[i,j], time_s[i,j] = s_v, chi2_v, e_v, np.mean(rad_e[i,j,:]), t_arr[i]
                                    except: pass
                        
                        # 【异常严格过滤】
                        valid_mask = (sif >= -5) & (sif <= 10) & (ka2 <= 1.2)
                        sif[~valid_mask] = np.nan
                        
                        m_sif = ~np.isnan(sif)
                        if np.any(m_sif):
                            col_indices = np.tile(np.arange(lons.shape[1]), (lons.shape[0], 1))
                            with nc.Dataset(out_nc, "w") as ods:
                                ods.createDimension("number", np.sum(m_sif))
                                ods.createDimension("wavelength_number", num_spectral)
                                ods.createVariable("pixel_index", "i4", ("number",))[:] = col_indices[m_sif]
                                ods.createVariable("SIF_740", "f4", ("number",))[:] = sif[m_sif]
                                ods.createVariable("SIF_err", "f4", ("number",))[:] = err[m_sif]
                                ods.createVariable("latitude", "f4", ("number",))[:] = lats[m_sif]
                                ods.createVariable("longitude", "f4", ("number",))[:] = lons[m_sif]
                                ods.createVariable("ka2", "f4", ("number",))[:] = ka2[m_sif]
                                ods.createVariable("Rad_mean", "f4", ("number",))[:] = rad_m[m_sif]
                                ods.createVariable("SZA", "f4", ("number",))[:] = SZA[m_sif]
                                ods.createVariable("VZA", "f4", ("number",))[:] = VZA[m_sif]
                                ods.createVariable("SAA", "f4", ("number",))[:] = SAA[m_sif]
                                ods.createVariable("VAA", "f4", ("number",))[:] = VAA[m_sif]
                                ods.createVariable("time", "f4", ("number",))[:] = time_s[m_sif]
                                
                                c_out = [cloud_fra_grid[np.argmin(np.abs(lat_sequence-lt)), np.argmin(np.abs(lon_sequence-ln))] for lt, ln in zip(lats[m_sif], lons[m_sif])]
                                ods.createVariable("cloud_fraction", "f4", ("number",))[:] = np.array(c_out)

                day_sif_files = f.get_nc_files(save_date_folder)
                if not day_sif_files: continue
                
                c_lats, c_lons, c_sif, c_err, c_ka2, c_cloud, c_rad, c_sza, c_vza, c_saa, c_vaa, c_time = [],[],[],[],[],[],[],[],[],[],[],[]
                for df in day_sif_files:
                    with nc.Dataset(df) as tmp_ds:
                        c_lats.append(tmp_ds['latitude'][:]); c_lons.append(tmp_ds['longitude'][:]); c_sif.append(tmp_ds['SIF_740'][:])
                        c_err.append(tmp_ds['SIF_err'][:]); c_ka2.append(tmp_ds['ka2'][:]); c_cloud.append(tmp_ds['cloud_fraction'][:])
                        c_rad.append(tmp_ds['Rad_mean'][:]); c_sza.append(tmp_ds['SZA'][:]); c_vza.append(tmp_ds['VZA'][:])
                        c_saa.append(tmp_ds['SAA'][:]); c_vaa.append(tmp_ds['VAA'][:]); c_time.append(tmp_ds['time'][:])
                
                cln, clt = np.concatenate(c_lons), np.concatenate(c_lats)
                grid_res = 0.5
                lon_seq, lat_seq = f.lon_lat_arange(grid_res)
                
                with nc.Dataset(day_nc, 'w') as dds:
                    dds.createDimension("lat", len(lat_seq)); dds.createDimension("lon", len(lon_seq))
                    dds.createVariable("latitude", "f4", ("lat",))[:] = lat_seq
                    dds.createVariable("longitude", "f4", ("lon",))[:] = lon_seq
                    
                    def save_grid(name, data_list):
                        grid = f.calculate_grid_mean(cln, clt, np.concatenate(data_list), grid_res)
                        dds.createVariable(name, "f4", ("lat", "lon"))[:] = grid

                    save_grid("SIF_740", c_sif); save_grid("SIF_err", c_err); save_grid("ka2", c_ka2)
                    save_grid("cloud_fraction", c_cloud); save_grid("Rad_mean", c_rad); save_grid("SZA", c_sza)
                    save_grid("VZA", c_vza); save_grid("SAA", c_saa); save_grid("VAA", c_vaa); save_grid("time", c_time)

print("🎉 Phase 2 (2003-2011) 极限求生反演全部完成！")
