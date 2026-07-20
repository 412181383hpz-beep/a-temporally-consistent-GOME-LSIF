import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
import os

# =====================================================
# 1. 配置参数
# =====================================================
SITES = ["Algeria1", "Algeria2", "Algeria3", "Algeria4", "Algeria5", "Libya1", "Mauritania1", "Mali1", "Niger1", "Niger2", "Niger3"]  
BASE_PATH = r"F:\python\site_nc_with_irr\station_1.5deg"
OUTPUT_BASE_DIR = r"E:\python\Degradation_correction\Results_Quartic_Fit"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 传感器受损前的分析 (1995-2003)
BREAK_DATE = None 
#BREAK_DATE = pd.to_datetime("2003-06-22")  
REF_DATE_1900 = pd.to_datetime("1900-01-01")  

# 定义像元类型映射 (0=East, 1=Nadir, 2=West)
PIXEL_TYPES = {0: "East", 1: "Nadir", 2: "West"}
PIXEL_COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"} # 蓝、绿、红

def seasonal_func(t, A, phi, offset):
    """用于去季节化的正弦函数"""
    return A * np.sin(2 * np.pi * t / 365.25 + phi) + offset

# =====================================================
# 主循环：逐站点处理
# =====================================================
for site in SITES:
    print(f"\n{'='*70}")
    print(f"📡 正在深度分析站点: {site} (四阶函数拟合)")
    
    PICS_FILE = fr"{BASE_PATH}\{site}_GOME_BAND4_199506_201107_1.5deg.nc"
    if not os.path.exists(PICS_FILE):
        print(f"跳过：找不到文件 {PICS_FILE}")
        continue

    # =====================================================
    # 2. 数据读取与维度对齐
    # =====================================================
    with xr.open_dataset(PICS_FILE, group="CALIBRATION") as ds_cal:
        wavelengths = ds_cal["wavelength"].values
    idx_758 = (np.abs(wavelengths - 758.0)).argmin()

    with xr.open_dataset(PICS_FILE, group="DAILY_IRRADIANCE") as ds_irr:
        irr_dates = pd.to_datetime(ds_irr["time"].values).normalize()
        daily_irrad_758 = np.nanmean(ds_irr["IRRADIANCE"].values[:, idx_758-1:idx_758+2], axis=1)
        irr_dict = dict(zip(irr_dates, daily_irrad_758))

    with xr.open_dataset(PICS_FILE, group="BAND4_FOOTPRINTS") as ds_fp:
        rad_raw = np.nanmean(ds_fp["radiance"].values[:, idx_758-1:idx_758+2], axis=1)
        
        def flatten_to_1d(arr):
            if arr.ndim > 1:
                return np.nanmean(arr, axis=tuple(range(1, arr.ndim)))
            return arr

        cloud_fraction = flatten_to_1d(ds_fp["cloud_fraction"].values)
        sza = flatten_to_1d(ds_fp["solar_zenith_angle"].values)
        vza = flatten_to_1d(ds_fp["viewing_zenith_angle"].values)
        pixel_indices = flatten_to_1d(ds_fp["source_pixel"].values).astype(int)

        time_data = ds_fp["time"].values
        fp_dates = pd.to_datetime(time_data).normalize()
        irrad_758 = np.array([irr_dict.get(d, np.nan) for d in fp_dates])

        cos_sza = np.cos(np.radians(sza))
        refl_758 = np.full_like(rad_raw, np.nan, dtype="float64")
        valid_mask = (cos_sza > 0) & (irrad_758 > 0) & np.isfinite(rad_raw)
        refl_758[valid_mask] = (np.pi * rad_raw[valid_mask]) / (cos_sza[valid_mask] * irrad_758[valid_mask])

    df_all = pd.DataFrame({
        "date": pd.to_datetime(time_data),
        "reflectance": refl_758,
        "pixel_type": pixel_indices,
        "cloud": cloud_fraction,
        "sza": sza,
        "vza": vza
    })

    # =====================================================
    # 3. 分像元循环拟合 (4阶多项式)
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13), sharex=True)
    
    for p_idx, p_name in PIXEL_TYPES.items():
        # 质量控制筛选
        mask = (df_all["pixel_type"] == p_idx) & (df_all["reflectance"] > 0) & \
               (df_all["cloud"] < 0.2) 
                #& (df_all["sza"] < 70) & (df_all["vza"] < 30)
        
        if BREAK_DATE is not None:
            mask = mask & (df_all["date"] < BREAK_DATE)

        df_p = df_all[mask].dropna().sort_values("date")
        if len(df_p) < 30: continue # 4阶拟合需要更多的数据点以保证稳定

        # --- 去季节化 ---
        t_days = (df_p["date"] - df_p["date"].min()).dt.days.values
        y_vals = df_p["reflectance"].values
        try:
            params, _ = optimize.curve_fit(seasonal_func, t_days, y_vals, p0=[np.std(y_vals), 0, np.mean(y_vals)])
            df_p["refl_deseason"] = y_vals - seasonal_func(t_days, params[0], params[1], 0)
            df_p_clean = df_p[np.abs(stats.zscore(y_vals - seasonal_func(t_days, *params))) < 2.0].copy()
        except:
            df_p_clean = df_p.copy()
            df_p_clean["refl_deseason"] = df_p_clean["reflectance"]

        # --- 核心：四阶多项式拟合 ---
        x_fit = (df_p_clean["date"] - REF_DATE_1900).dt.days.values / 10000.0
        y_fit = df_p_clean["refl_deseason"].values
        
        # 将阶数设为 4
        coeffs = np.polyfit(x_fit, y_fit, 4)
        poly = np.poly1d(coeffs)
        y_pred = poly(x_fit)
        
        # 统计计算
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        fit_count = len(df_p_clean)

        # 构造 4 阶方程字符串 (ax^4 + bx^3 + cx^2 + dx + e)
        a, b, c, d, e = coeffs
        eq_label = (f"{p_name}: $y = {a:.3f}x^4 {b:+.3f}x^3 {c:+.3f}x^2 {d:+.3f}x {e:+.3f}$\n"
                    f"($R^2={r_squared:.4f}, N={fit_count}$)")
        
        # --- 绘图 ---
        color = PIXEL_COLORS[p_idx]
        ax1.scatter(df_p_clean["date"], df_p_clean["refl_deseason"], color=color, s=5, alpha=0.1, label='_nolegend_')
        ax1.plot(df_p_clean["date"], y_pred, color=color, linewidth=2.5, label=eq_label)
        
        # Dfactor
        df_p_clean["Dfactor"] = y_pred / poly(x_fit[0])
        ax2.plot(df_p_clean["date"], df_p_clean["Dfactor"], color=color, linewidth=2.5, label=f"{p_name}")

    # 4. 图表细节美化
    ax1.set_title(f"GOME-1 758nm Degradation: {site} (4th-Order Polynomial Fit)", fontsize=14)
    ax1.set_ylabel("TOA Reflectance (Deseasonalized)", fontsize=12)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, framealpha=0.8) # 将长图例移到图外或调小
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.set_title("Relative Degradation Factor (Dfactor)", fontsize=14)
    ax2.set_ylabel("Dfactor", fontsize=12)
    ax2.set_xlabel("Time (Year)", fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, f"{site}_Quartic_Stats_Analysis.png"), dpi=300)
    plt.close()
    print(f"✅ {site} 处理完成，4阶拟合 R²={r_squared:.4f}")

print(f"\n{'='*70}\n🎉 四阶函数拟合分析已全部完成！")