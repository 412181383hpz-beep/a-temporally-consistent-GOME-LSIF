import os
import numpy as np
from datetime import datetime
from glob import glob

# =================================================================
# 配置区域：分像元四阶多项式系数 (已扩展为5个参数)
# =================================================================
# 请将你拟合终端输出的各像元 4 阶衰减系数 [a, b, c, d, e] 填入此处
# 对应公式: y = ax^4 + bx^3 + cx^2 + dx + e
# 0: East, 1: Nadir (Center), 2: West
PIXEL_COEFFS = {
    0: [-11.248, 173.150, -998.196, 2554.176, -2446.991],  # East FOV
    1: [-12.326, 186.948, -1062.173, 2679.343, -2531.249],  # Nadir/centre FOV
    2: [-2.160, 32.928, -188.111, 477.258, -453.152]   # West FOV
}

def get_dfactor(current_date, pixel_type):
    """
    根据给定的观测日期和像元类型计算衰减因子 Dfactor。
    公式: Dfactor(NOD) = P(NOD) / P(NOD_baseline)
    """
    coeffs = PIXEL_COEFFS.get(pixel_type, PIXEL_COEFFS[1])
    poly_func = np.poly1d(coeffs)
    
    # 定义 1900年1月1日 为绝对基准
    ref_date_1900 = datetime(1900, 1, 1)
    
    # 定义未衰减的物理基准时间 (baseline)
    baseline_date = datetime(1995, 7, 1) 
    baseline_nod = (baseline_date - ref_date_1900).days
    
    current_nod = (current_date - ref_date_1900).days
    
    # 【已修正】统一缩放因子为 10000.0，与拟合脚本保持绝对一致
    x_current = current_nod / 10000.0
    x_baseline = baseline_nod / 10000.0
    
    rad_baseline = poly_func(x_baseline)
    y_current = poly_func(x_current)
    
    dfactor = y_current / rad_baseline
    return dfactor

# 【已修正】扩展年份至 2011 年
years = list(range(1995, 2012)) 
base_in_dir = r'F:\traindata_GOME_Band4_NADIR'
base_out_dir = r'traindata_GOME_Band4_corrected_fourthorder'

for year in years:
    in_dir = os.path.join(base_in_dir, str(year))
    out_dir = os.path.join(base_out_dir, str(year))
    os.makedirs(out_dir, exist_ok=True)
    
    # 如果某一年文件夹不存在，跳过
    if not os.path.exists(in_dir):
        print(f"目录 {in_dir} 不存在，跳过该年份。")
        continue

    npz_files = glob(os.path.join(in_dir, '*_traindata.npz'))
    
    for file_path in npz_files:
        file_name = os.path.basename(file_path)
        date_str = file_name.split('_')[0]
        current_date_obj = datetime.strptime(date_str, "%Y%m%d")
        
        data = np.load(file_path)
        raw_traindata = data['traindata']
        traincols = data['traincols']
        
        corrected_traindata = np.copy(raw_traindata)
        
        # 核心：根据像元类型分流计算 Dfactor 并进行四阶校正
        for p_type in [0, 1, 2]:
            mask = (traincols == p_type)
            if np.any(mask):
                dfactor = get_dfactor(current_date_obj, p_type)
                corrected_traindata[mask] = raw_traindata[mask] / dfactor
        
        out_path = os.path.join(out_dir, file_name)
        np.savez(out_path,
                 traindata=corrected_traindata,
                 traincols=data['traincols'],
                 traintype=data['traintype'],
                 trainlons=data['trainlons'],
                 trainlats=data['trainlats'],
                 trainsza=data['trainsza'])
        
    print(f"{year} 年的训练集 Dfactor 分像元校正并转存完毕！")
