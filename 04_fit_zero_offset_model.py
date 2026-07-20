import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from scipy.special import eval_legendre
from datetime import datetime
import gome_sif_utils as f

# =================================================================
# 1. 配置参数
# =================================================================
YEARS = range(1995, 2012)  # 包含 1995 到 2011
MONTHS = range(1, 13)

# 路径根据您的两个阶段自动切换
TRAIN_ROOT = r'F:\traindata_GOME_Band4_corrected_fourthorder'
# 注意：这里需要指向对应阶段的反演结果目录
BASIS_ROOT_N4 = r'F:\SIF740_GOME_N4_corrected_FourthOrder' # 1995-2003.6
BASIS_ROOT_N4_2011 = r'F:\SIF740_GOME_N4_corrected_FourthOrder'                   # 2003.7-2011

OUTPUT_CSV = r'E:\python\SIF反演算法\Zero_Level_Offset_Correction\Zero_Bias_Factors_corrected_FourthOrder.csv'

# 光谱配置
start_wl, end_wl = 743, 758
num_spectral = 72
npc, npp = 4, 3
num_column = 3

wl_crop = np.linspace(start_wl, end_wl, num_spectral)
wn = 2 * (wl_crop - np.min(wl_crop)) / (np.max(wl_crop) - np.min(wl_crop)) - 1
hf = np.exp(-(wl_crop - 740)**2 / (2 * 10**2))

# =================================================================
# 2. 辅助函数：建立基函数映射表
# =================================================================
def build_basis_map(year_dir):
    """
    扫描文件夹下所有基函数，返回一个排序好的 (开始日期int, 文件路径) 列表
    """
    files = glob.glob(os.path.join(year_dir, "SVD_Basis_*.npz"))
    if not files: return []
    
    basis_meta = []
    for f_path in files:
        # 提取文件名中的日期 SVD_Basis_20030701.npz -> 20030701
        try:
            d_int = int(os.path.basename(f_path).split('_')[2].split('.')[0])
            basis_meta.append((d_int, f_path))
        except: continue
        
    return sorted(basis_meta, key=lambda x: x[0])

def get_correct_basis(current_date_int, basis_map):
    """
    寻找小于或等于当前日期的最大基函数日期点
    """
    best_match = None
    for d_int, f_path in basis_map:
        if current_date_int >= d_int:
            best_match = f_path
        else:
            break
    return best_match

# =================================================================
# 3. 主循环
# =================================================================
results = []

for year in YEARS:
    # 策略判断：2003年7月之前使用 N4 目录，之后使用 N4_2011 目录
    basis_year_dir = BASIS_ROOT_N4_2011 if year >= 2004 else BASIS_ROOT_N4
    # 2003年需要特殊处理月份（此处简化，实际应用中可加入月度判断）
    
    basis_map = build_basis_map(os.path.join(basis_year_dir, str(year)))
    
    for month in MONTHS:
        print(f"\n分析周期: {year}-{month}")
        month_str = str(month).zfill(2)
        train_files = glob.glob(os.path.join(TRAIN_ROOT, str(year), f"{year}{month_str}*_traindata.npz"))
        if not train_files: continue

        # 按天收集数据，每读取一个文件都动态加载其对应的基函数
        m_rad, m_sza, m_lat, m_col, m_bias = [], [], [], [], []

        for t_file in tqdm(train_files, desc="同步基函数反演"):
            date_int = int(os.path.basename(t_file).split('_')[0])
            
            # --- 关键：寻找该光谱对应的基函数版本 ---
            basis_path = get_correct_basis(date_int, basis_map)
            if not basis_path: continue
            
            # 加载并重建基矩阵
            svd_data = np.load(basis_path)
            V_all = svd_data['V_all']
            poly_all = np.zeros((num_column, num_spectral, npc + npp + 1))
            for c_i in range(num_column):
                if np.isnan(V_all[c_i, 0, 0]): continue
                pm = np.zeros((num_spectral, npc + npp))
                for p in range(npc): pm[:, p] = V_all[c_i][p, :]
                for p in range(npp): pm[:, p + npc] = eval_legendre(p + 1, wn) * V_all[c_i][0, :]
                poly_all[c_i] = np.hstack((pm, hf[:, np.newaxis]))

            # 读取训练集光谱并反演
            data = np.load(t_file)
            if 'trainsza' not in data: continue # 确保补丁已打
            
            spectra = data['traindata']
            szas, lats, cols = data['trainsza'], data['trainlats'], data['traincols']
            
            for i in range(len(spectra)):
                c_idx = int(cols[i])
                try:
                    # 使用同步后的基函数反演初始 Bias
                    s_v, chi2_v, e_v = f.sif_retrieval(poly_all[c_idx], spectra[i], npc+npp+1)
                    if not np.isnan(s_v) and -10 < s_v < 10 and chi2_v < 5:
                        m_bias.append(s_v)
                        m_rad.append(np.mean(spectra[i]))
                        m_sza.append(szas[i])
                        m_lat.append(lats[i])
                        m_col.append(c_idx)
                except: continue

        # 执行 4 参数线性拟合 (A, B, C, D)
        m_bias, m_sza, m_rad, m_lat, m_col = map(np.array, [m_bias, m_sza, m_rad, m_lat, m_col])
        valid = np.isfinite(m_sza) & np.isfinite(m_rad) & np.isfinite(m_bias)
        
        for c_idx in range(num_column):
            mask = (m_col == c_idx) & valid
            if np.sum(mask) < 500: continue
            
            # 构建 X 矩阵: [1, SZA, Rad, Lat]
            Y = m_bias[mask] / np.cos(np.deg2rad(m_sza[mask]))
            X = np.column_stack((np.ones_like(m_sza[mask]), m_sza[mask], m_rad[mask], m_lat[mask]))
            
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            
            results.append({
                'Year': year, 'Month': month, 'Column': c_idx, 'Sample_Size': np.sum(mask),
                'A': coeffs[0], 'B': coeffs[1], 'C': coeffs[2], 'D': coeffs[3]
            })

# 保存
if results:
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"拟合完成，同步因子已存入: {OUTPUT_CSV}")
