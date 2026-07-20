import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from tqdm import tqdm

# =================================================================
# 1. 核心配置区域
# =================================================================
# 全生命周期同步校正因子表
CSV_PATH = r'E:\python\SIF反演算法\Zero_Level_Offset_Correction\Zero_Bias_Factors_corrected_FourthOrder.csv'

# ★ 更新：现在所有 1995-2011 年的数据统一在此根目录下 ★
SIF_ROOTS = [
    r'F:\SIF740_GOME_N4_corrected_FourthOrder'
]

# 读取并缓存校正因子
try:
    df_factors = pd.read_csv(CSV_PATH)
    print(f"✅ 成功加载校正因子表，共 {len(df_factors)} 条记录。")
except Exception as e:
    print(f"❌ 读取 CSV 失败: {e}")
    exit()

# =================================================================
# 2. 核心校正函数 (极速追加模式)
# =================================================================
def apply_correction_to_file(nc_file, year, month):
    """
    计算并在单日 SIF.nc 文件中追加校正变量
    """
    # 提取该年该月的 3 列校正系数
    monthly_factors = df_factors[(df_factors['Year'] == year) & (df_factors['Month'] == month)]
    if monthly_factors.empty:
        return False, "因子缺失"

    try:
        # 以 'a' (append) 模式打开，直接在原文件上修改
        with nc.Dataset(nc_file, 'a') as ds:
            # 提取参与计算的基础变量
            sza = ds.variables['SZA'][:]
            rad = ds.variables['Rad_mean'][:]
            lat = ds.variables['latitude'][:]
            pix_idx = ds.variables['pixel_index'][:]
            sif_raw = ds.variables['SIF_740'][:]
            
            # 初始化偏差数组
            num_pixels = len(sif_raw)
            bias_array = np.zeros(num_pixels, dtype='f4')
            
            # 分 3 个探测器列分别计算偏差
            for c_idx in [0, 1, 2]:
                mask = (pix_idx == c_idx)
                if not np.any(mask):
                    continue
                
                # 获取该列的 A, B, C, D
                row = monthly_factors[monthly_factors['Column'] == c_idx]
                if row.empty:
                    continue
                
                A = row['A'].values[0]
                B = row['B'].values[0]
                C = row['C'].values[0]
                D = row['D'].values[0]
                
                # 核心公式: bias = cos(SZA) * (A + B*SZA + C*Rad + D*lat)
                cos_sza = np.cos(np.deg2rad(sza[mask]))
                bias_val = cos_sza * (A + B * sza[mask] + C * rad[mask] + D * lat[mask])
                bias_array[mask] = bias_val
                
            # 执行物理校正
            sif_corrected = sif_raw - bias_array
            
            # ================= 写入 / 更新变量 =================
            def write_or_update_var(var_name, data, long_name):
                if var_name not in ds.variables:
                    v = ds.createVariable(var_name, 'f4', ('number',))
                    v.units = 'mW/m2/nm/sr'
                    v.long_name = long_name
                else:
                    v = ds.variables[var_name]
                v[:] = data

            write_or_update_var('SIF_bias', bias_array, 'Zero-level offset bias calculated from synchronous linear model')
            write_or_update_var('SIF_740_corrected', sif_corrected, 'SIF at 740nm corrected for zero-level offset')

        return True, ""
    except Exception as e:
        return False, str(e)

# =================================================================
# 3. 超长序列全自动遍历执行
# =================================================================
print("\n=== 🚀 开始全生命周期 SIF 零偏差校正追加任务 ===")

total_processed = 0
total_skipped = 0

for root_dir in SIF_ROOTS:
    if not os.path.exists(root_dir):
        print(f"⚠️ 找不到目录: {root_dir}，已跳过。")
        continue
        
    print(f"\n📂 正在扫描目录: {root_dir}")
    # 自动同时匹配 199x 年和 20xx 年的文件夹
    year_folders = sorted(glob.glob(os.path.join(root_dir, "20*")) + glob.glob(os.path.join(root_dir, "19*")))
    
    for y_dir in year_folders:
        year = int(os.path.basename(y_dir))
        day_folders = sorted([d for d in glob.glob(os.path.join(y_dir, '*')) if os.path.isdir(d)])
        
        if not day_folders:
            continue
            
        # 使用 tqdm 在控制台打印年度进度条
        for d_dir in tqdm(day_folders, desc=f"处理 {year} 年数据"):
            month_str = os.path.basename(d_dir)[4:6]
            month = int(month_str)
            
            # 获取该日期的所有 SIF 单轨文件
            sif_files = glob.glob(os.path.join(d_dir, '*_SIF.nc'))
            for f_path in sif_files:
                success, error_msg = apply_correction_to_file(f_path, year, month)
                if success:
                    total_processed += 1
                else:
                    total_skipped += 1

print("\n🎉 全部校正完毕！")
print(f"✅ 成功追加变量的文件数: {total_processed}")
print(f"⚠️ 缺失因子或出错跳过数: {total_skipped}")
