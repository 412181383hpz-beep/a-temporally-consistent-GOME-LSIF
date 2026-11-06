# GOME-LSIF 代码运行说明

## 环境准备

在仓库根目录打开终端，使用Python 3.10或更高版本：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt`列出直接依赖，`requirements-verified.txt`记录此前验证时使用的已安装版本。后者不是完整环境锁定文件。

## 准备外部输入

仓库提供处理代码与两张生产参考表。完整GOME Level 1b、每日AVHRR LTDR及Mali1站点数据需要另行准备。

默认目录为：

```text
data/
  GOME/YYYY/MM/DD/<GOME Level 1b文件>
  AVHRR/YYYY/<每日AVHRR NetCDF文件>
  PICS/Mali1_GOME_BAND4_199506_201107_1.5deg.nc
```

AVHRR输入应包含`SREFL_CH1`、`SREFL_CH2`、`BT_CH4`。Mali1输入需要`CALIBRATION`、`DAILY_IRRADIANCE`、`BAND4_FOOTPRINTS`分组。具体输入要求见README。也可在各脚本顶部修改输入输出路径。

## 运行顺序

始终从仓库根目录运行；相对路径以当前工作目录为基准：

```powershell
python 00_fit_pics_degradation_model.py
python 01_select_nonfluorescent_training_samples.py
python 02_correct_training_spectra_for_degradation.py
python 03a_retrieve_sif_1995_2003.py
python 03b_retrieve_sif_2003_2011.py
python 04_fit_zero_offset_model.py
python 05_apply_zero_offset_correction.py
python 06_generate_public_product.py
```

默认输出分别写入`outputs/degradation`、`training_raw`、`training_corrected`、`pre_zero`、`zero_offset`及`GOME_LSIF_public_v1_0`，均位于`outputs/`下面。

00输出拟合表和诊断；02及共用反演模块使用已经核对的固定Mali1二阶系数，不自动采用每次00的新拟合结果。季节Fourier项不应用到单条光谱的退化校正。

03a覆盖1995-07-01至2003-06-30，03b覆盖2003-07-01至2011-06-30；两个入口都依赖同目录下的`retrieve_gome_sif_orbit_Mali1_quadratic_S2_SZA75.py`。训练选择脚本还依赖`gome_sif_utils.py`。

完整可用训练日期清单决定分组。完成训练数据准备后，可预览任务：

```powershell
python 03a_retrieve_sif_1995_2003.py --dry-run
python 03b_retrieve_sif_2003_2011.py --dry-run
```

完整反演完成后再运行全时期04拟合。检查其系数manifest的`status=PASS`且`blocked_tasks`为空，然后运行05。06生成的本次`run_summary.json`为PASS，才代表本次打包完成。

## 生产参考表

- [production_task_schedule.csv](reference/production_task_schedule.csv)：567个生产周期的编号和完整训练、反演日期。
- [production_zero_offset_coefficients.csv](reference/production_zero_offset_coefficients.csv)：1,701组“周期×视场”的零偏差系数及拟合样本信息。

两张表用于核对生产配置。主流程会建立并检查本次任务表、重新拟合系数；05依赖04生成的系数和完整来源manifest，不能只用参考CSV替代这些运行输入。

## 关键设置与输出

- 每个周期、每个视场分别要求有效候选至少100条、抽样后至少100条、PCA后至少72条；后期200条是补样目标。
- 训练SZA <75°、VZA <60°；使用Mali1二阶长期退化校正，以及cosSZA_cosLat七参数二阶零偏差模型，无交互项。
- 源中间产品SIF为[-5,10]；共享筛选使用零偏差校正前SIF [-5,5]和退化校正后平均辐亮度[25,200]，均含端点，单位为mW m-2 sr-1 nm-1。
- 共享筛选不额外筛云，不对最终校正后SIF增加范围筛选；保留空轨道。
- 推荐科学变量为`sif_740_zero_offset_corrected`；`sif_740`为零偏差校正前SIF，`sif_bias`为被减去的估计偏差。
- `sif_standard_error`描述理论反演精度，不包含系统误差和校正模型的不确定性。

代码已通过合成个案及选定真实任务的一致性检查，未重跑全时段。运行生成的逐轨道记录和报告保存在本次输出目录中。
