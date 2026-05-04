# 方案B：UNet + MobileNetV3-Small 金属零件缺陷分割 —— 从零到部署完全手册

---

## 1. 方案简介

本项目基于 **STM32MP157DAA1** 工业缺陷检测与分拣系统，检测金属零件（垫圈、冲压件等）表面缺陷。

选择的方案是 **UNet + MobileNetV3-Small** 分割：

```
输入：零件表面图像 (224x224)
      │
      ▼
┌─────────────────────────┐
│  Encoder（收缩路径）       │  ← MobileNetV3-Small 预训练backbone
│  224→112→56→28→14        │     逐层提取特征，空间越来越小
└─────────────────────────┘
      │
      ├──── Skip Connection 1 (112x112)  ─────────┐
      ├──── Skip Connection 2 (56x56)   ─────────┐ │
      ├──── Skip Connection 3 (28x28)   ────────┐ │ │
      ├──── Skip Connection 4 (14x14)   ──────┐ │ │ │
      │                                       │ │ │ │
      ▼                                       ▼ ▼ ▼ ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  Bottleneck (7x7)       │  ──→  │  Decoder（扩展路径）       │
│  最深层特征              │       │  14→28→56→112→224        │
└─────────────────────────┘       │  每层拼接Skip特征          │
                                  │  恢复空间分辨率             │
                                  └─────────────────────────┘
                                            │
                                            ▼
                                   像素级分割掩码 (224x224)
                                   每个像素：背景/划痕/锈蚀/裂纹/...
```

**为什么选这个方案**：
- UNet 的 Skip Connection 能保留细小缺陷（细裂纹、小压伤）的边界信息
- MobileNetV3-Small backbone 保证模型足够轻量，能在 STM32MP157 上跑
- 预训练 ImageNet 权重加速收敛，小数据集也能训练

---

## 2. 环境准备（已完成）

### 2.1 本机硬件配置

| 项目 | 实际配置 |
|---|---|
| 操作系统 | Windows 11 Home |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8GB显存） |
| GPU驱动 | 555.97 |
| CUDA Driver | 12.5（nvidia-smi显示） |
| 内存 | 16GB |
| CPU | 笔记本处理器 |

### 2.2 软件环境（已安装）

| 组件 | 版本 | 安装位置 |
|---|---|---|
| Miniconda | 25.3.1 | `E:\Miniconda` |
| Python | 3.9 | `D:\model_picture\defect-unet` |
| PyTorch | 2.6.0+cu124 | 同上（conda环境内） |
| SMP | 0.5.0 | 同上 |
| CUDA Runtime | 12.4 | PyTorch内嵌（兼容12.5驱动） |

**CUDA兼容说明**：nvidia-smi显示的CUDA 12.5是驱动版本，PyTorch自带的CUDA 12.4是运行时版本。驱动版本≥运行时版本即可正常工作，12.5 > 12.4，完全兼容。

### 2.3 每次开机激活环境

#### 2.3.1 VSCode PowerShell（推荐）

你的 PowerShell profile 已经加载了 Conda hook，因此在 VSCode 中通常可以直接执行：

```powershell
# 第1步：进入项目目录
cd D:\model_picture

# 第2步：激活训练环境
conda activate D:\model_picture\defect-unet

# 第3步：确认当前python来自训练环境
python -c "import sys; print(sys.executable)"
```

正确输出应为：

```text
D:\model_picture\defect-unet\python.exe
```

如果某次新终端提示 `conda` 找不到，先手动加载一次 Conda hook：

```powershell
& "E:\Miniconda\shell\condabin\conda-hook.ps1"
conda activate D:\model_picture\defect-unet
```

#### 2.3.2 CMD 终端

如果打开的是 CMD，不是 PowerShell，则执行：

```bat
E:\Miniconda\condabin\conda_hook.bat
conda activate D:\model_picture\defect-unet
cd /d D:\model_picture
```

激活成功后命令行前缀通常会出现 `(D:\model_picture\defect-unet)`。

#### 2.3.3 VSCode中激活后常用命令

| 任务 | 命令 |
|---|---|
| 检查当前Python | `python -c "import sys; print(sys.executable)"` |
| 检查PyTorch/GPU | `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` |
| 快速测试训练 | `python train.py --data_dir D:\model_picture\datasets_severstal --num_classes 5 --epochs 2 --batch_size 4 --num_workers 0 --log_interval 10` |
| 正式训练 | `python train.py --data_dir D:\model_picture\datasets_severstal --num_classes 5 --epochs 100 --batch_size 4 --num_workers 0 --log_interval 50` |
| 中断后续训 | `python train.py --data_dir D:\model_picture\datasets_severstal --num_classes 5 --epochs 100 --batch_size 4 --num_workers 0 --resume auto --log_interval 50` |
| 启动TensorBoard | `tensorboard --logdir D:\model_picture\logs` |
| 导出ONNX | `python export_onnx.py --checkpoint D:\model_picture\checkpoints\best_model.pth --output D:\model_picture\checkpoints\defect_unet.onnx --num_classes 5 --encoder mobilenet_v2` |
| 检查ONNX文件 | `Get-Item D:\model_picture\checkpoints\defect_unet.onnx` |
| 电脑端UVC实时推理 | `python infer_camera_onnx.py --camera_id 1 --model D:\model_picture\checkpoints\defect_unet.onnx --num_classes 5` |

最短使用流程：

```powershell
cd D:\model_picture
conda activate D:\model_picture\defect-unet
python train.py --data_dir D:\model_picture\datasets_severstal --num_classes 5 --epochs 2 --batch_size 4 --num_workers 0 --log_interval 10
```

训练完成后导出：

```powershell
python export_onnx.py --checkpoint D:\model_picture\checkpoints\best_model.pth --output D:\model_picture\checkpoints\defect_unet.onnx --num_classes 5 --encoder mobilenet_v2
```

### 2.4 验证环境

```bash
# 检查conda
conda --version
# 应输出: conda 25.3.1

# 检查GPU
nvidia-smi
# 应显示 RTX 4060 Laptop, CUDA 12.5

# 检查PyTorch + GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 应输出: PyTorch 2.6.0+cu124, CUDA: True, GPU: NVIDIA GeForce RTX 4060 Laptop GPU

# 检查SMP
python -c "import segmentation_models_pytorch as smp; print(f'SMP {smp.__version__}')"
# 应输出: SMP 0.5.0
```

---

## 3. 项目目录结构

项目根目录：`D:\model_picture`

```
D:\model_picture\
├── defect-unet\              ← Conda虚拟环境（Python 3.9 + PyTorch）
├── datasets\                 ← 训练数据（转换后）
│   ├── train\
│   │   ├── images\           ← 训练原图（70%）
│   │   └── masks\            ← 训练掩码 PNG（70%）
│   ├── val\
│   │   ├── images\           ← 验证原图（15%）
│   │   └── masks\            ← 验证掩码 PNG（15%）
│   └── test\
│       ├── images\           ← 测试原图（15%）
│       └── masks\            ← 测试掩码 PNG（15%）
├── raw_images\               ← Labelme原始标注JSON + 原图
├── converted_images\         ← 转换后的原图（convert_labelme.py生成）
├── converted_masks\          ← 转换后的掩码（convert_labelme.py生成）
├── checkpoints\              ← 模型权重保存
│   ├── best_model.pth        ← 最佳模型
│   └── defect_unet.onnx      ← 导出的ONNX模型
├── logs\                     ← TensorBoard日志
├── dataset.py                ← 数据集加载代码（含数据增强定义）
├── train.py                  ← 训练脚本（含参数说明）
├── export_onnx.py            ← ONNX导出脚本
├── infer_camera_onnx.py      ← 电脑端UVC摄像头ONNX实时推理脚本
├── convert_labelme.py        ← Labelme标注转换 + 数据集划分脚本
├── capture.py                ← UVC摄像头截图脚本
└── 操作手册.md               ← 操作手册
```

### 3.1 核心训练代码说明

所有Python文件已放在 `D:\model_picture` 下，代码带详细注释，此处不再重复源码。各文件功能：

| 文件 | 功能 | 关键说明 |
|---|---|---|
| `dataset.py` | 数据集加载+数据增强 | DefectDataset类、训练/验证增强、ImageNet归一化 |
| `train.py` | 主训练脚本 | UNet+轻量Encoder、CE+Dice联合损失、AdamW、余弦退火、batch级进度日志 |
| `export_onnx.py` | 导出ONNX | .pth→.onnx，验证模型结构 |
| `infer_camera_onnx.py` | 电脑端实时推理 | ONNX Runtime + OpenCV，UVC摄像头实时显示分割结果 |
| `convert_labelme.py` | 标注转换+划分 | JSON→掩码PNG、train/val/test划分 |
| `capture.py` | 摄像头截图 | UVC摄像头640x480采集、亮度调节 |

---

## 4. 数据采集与标注操作步骤

### 4.1 拍摄零件照片

**摄像头**：UVC摄像头，分辨率640x480

使用 `capture.py` 脚本采集：

```bash
conda activate D:\model_picture\defect-unet
cd /d D:\model_picture

# 拍良品样本
python capture.py --save_dir ./raw_images/good --camera_id 0

# 拍划痕样本
python capture.py --save_dir ./raw_images/scratch --camera_id 0

# 拍锈蚀样本
python capture.py --save_dir ./raw_images/rust --camera_id 0

# 拍压伤样本
python capture.py --save_dir ./raw_images/dent --camera_id 0

# 拍裂纹样本
python capture.py --save_dir ./raw_images/crack --camera_id 0

# 拍毛刺样本
python capture.py --save_dir ./raw_images/burr --camera_id 0
```

`capture.py` 操作按键：
- **s** = 截图保存
- **q** = 退出
- **+** = 增加亮度（模拟强光条件，增加数据多样性）
- **-** = 降低亮度（模拟弱光条件）
- **0** = 恢复原始亮度

`--camera_id` 参数：0=第一个摄像头，外接UVC可能是1或2，黑屏就换编号。

**拍摄要求**：

| 要求 | 说明 |
|---|---|
| 用实际工位的UVC摄像头拍 | 不要用手机，镜头畸变和色彩不同 |
| 用实际工位的光源 | 补光角度、色温必须一致 |
| 分辨率640x480 | 你的UVC摄像头默认分辨率 |
| 零件居中、固定焦距 | 手动对焦后锁定 |
| 每张图只有一个零件 | 避免多零件干扰 |

**数量要求**：

| 类别 | 最少 | 推荐 | 理想 |
|---|---|---|---|
| 良品 | 500张 | 1000张 | 2000+张 |
| 每类缺陷 | 150张 | 300-500张 | 500+张 |

### 4.2 用Labelme标注

```bash
conda activate D:\model_picture\defect-unet
cd /d D:\model_picture
labelme ./raw_images --labels scratch,rust,dent,crack,burr
```

Labelme操作步骤（每张图片）：
1. 左侧选择一张图片
2. 右侧点击 "Create Polygons"（创建多边形）
3. 沿缺陷边缘点击画多边形，双击结束
4. 在弹出对话框中选择标签名（如 scratch）
5. 有多个缺陷就重复步骤2-4
6. 良品图片直接保存（空标注 = 全背景）
7. File → Save 保存JSON文件

标注后每个图片生成一个同名JSON文件：

```
raw_images/
├── good/
│   ├── 0000.jpg
│   ├── 0000.json
│   ├── 0001.jpg
│   ├── 0001.json
│   └── ...
├── scratch/
│   ├── 0000.jpg
│   ├── 0000.json
│   └── ...
└── ...
```

**标注质量要求**：

| 要求 | 说明 |
|---|---|
| 边界贴合缺陷 | 不要画太大也不要画太小 |
| 同一缺陷一个标注 | 不要把一个缺陷拆成多个多边形 |
| 良品也要保存JSON | 空标注=全背景，防止误检 |
| 标签名必须严格匹配 | 只能用 scratch/rust/dent/crack/burr |

### 4.3 转换标注并划分数据集

```bash
# 步骤1：将所有子目录的Labelme JSON合并转换
# 需要把各子目录的JSON都集中转换
python convert_labelme.py convert ./raw_images/good ./converted_images ./converted_masks
python convert_labelme.py convert ./raw_images/scratch ./converted_images ./converted_masks
python convert_labelme.py convert ./raw_images/rust ./converted_images ./converted_masks
python convert_labelme.py convert ./raw_images/dent ./converted_images ./converted_masks
python convert_labelme.py convert ./raw_images/crack ./converted_images ./converted_masks
python convert_labelme.py convert ./raw_images/burr ./converted_images ./converted_masks

# 步骤2：划分为train/val/test（70%/15%/15%）
python convert_labelme.py split ./converted_images ./converted_masks ./datasets
```

划分后目录结构：

```
D:\model_picture\datasets\
├── train\
│   ├── images\     ← 70% 的原图
│   └── masks\      ← 70% 的掩码
├── val\
│   ├── images\     ← 15% 的原图
│   └── masks\      ← 15% 的掩码
└── test\
    ├── images\     ← 15% 的原图
    └── masks\      ← 15% 的掩码
```

---

## 5. 开始训练

### 5.1 标准训练命令

```bash
# 激活环境
E:\Miniconda\condabin\conda_hook.bat
conda activate D:\model_picture\defect-unet
cd /d D:\model_picture

# 开始训练（默认参数）
python train.py --data_dir ./datasets --epochs 100 --batch_size 8
```

### 5.2 在VSCode中训练

1. 打开VSCode → 文件 → 打开文件夹 → `D:\model_picture`
2. Ctrl+` 打开终端
3. 执行：
   ```powershell
   cd D:\model_picture
   conda activate D:\model_picture\defect-unet
   python -c "import sys; print(sys.executable)"
   python train.py --data_dir D:\model_picture\datasets_severstal --num_classes 5 --epochs 2 --batch_size 4 --num_workers 0 --log_interval 10
   ```

如果 `conda` 提示找不到，先执行：

```powershell
& "E:\Miniconda\shell\condabin\conda-hook.ps1"
conda activate D:\model_picture\defect-unet
```

或配置launch.json后直接按F5运行（详见操作手册.md第11节）。

### 5.3 显存不够时的调整

| 现象 | 解决方法 | 命令 |
|---|---|---|
| CUDA Out of Memory | 减小batch_size | `--batch_size 4` 或 `--batch_size 2` |
| 4也不够 | 减小batch_size到2 + 梯度累积 | 修改train.py加梯度累积 |
| 还是不够 | 用更小的输入尺寸 | 修改dataset.py中 Resize 为 160x160 |

你的 RTX 4060 8GB 显存，batch_size=8 大约占4-5GB，没有问题。

### 5.4 训练参数详解

```bash
python train.py \
    --data_dir ./datasets \          # 数据集目录（由convert_labelme.py split生成）
    --num_classes 6 \                # 类别数：背景+5种缺陷
    --batch_size 8 \                # 批大小：RTX 4060可用8，显存不够改4或2
    --epochs 100 \                  # 训练轮数：先跑100轮看效果
    --lr 1e-3 \                     # 学习率：UNet收敛快，1e-3合适
    --encoder mobilenet_v2 \        # 当前环境可直接运行的轻量backbone
    --checkpoint_dir ./checkpoints \# 模型保存目录
    --log_dir ./logs \              # TensorBoard日志目录
    --num_workers 4 \               # 数据加载线程（报BrokenPipeError改0）
    --log_interval 50 \             # 每50个batch打印一次训练/验证进度
    --resume "" \                   # 空字符串=从头训练；auto=从last_checkpoint续训
    --patience 12 \                 # 验证mIoU连续12轮不提升则提前停止
    --seed 42 \                     # 固定随机种子，方便复现实验
    --class_weights auto            # 自动估计类别权重，缓解背景/缺陷类别不平衡
```

### 5.4.1 训练轮数、续训和最佳模型

训练不是每一轮重新随机开始，而是连续衔接：

```text
初始化模型
  ↓
Epoch 1 更新当前模型权重
  ↓
Epoch 2 接着 Epoch 1 的权重继续更新
  ↓
Epoch 3 接着 Epoch 2 的权重继续更新
```

| 文件 | 作用 | 什么时候用 |
|---|---|---|
| `best_model.pth` | 验证集mIoU历史最高的模型 | 最终导出ONNX、部署时优先使用 |
| `last_checkpoint.pth` | 每个epoch结束都保存的最新训练状态 | 训练被 `Ctrl+C` 或意外中断后续训 |
| `checkpoint_epoch20.pth` | 每20轮保存一次的阶段快照 | 长训练时保留阶段性备份 |

`--epochs` 表示目标总轮数：

| 场景 | 命令含义 |
|---|---|
| `--epochs 100` 从头训练 | 从第1轮训练到第100轮 |
| `--resume auto --epochs 100` | 从 `last_checkpoint.pth` 记录的下一轮继续，直到第100轮结束 |
| checkpoint已到第5轮，执行 `--resume auto --epochs 20` | 从第6轮继续训练到第20轮 |

中断后继续训练：

```powershell
python train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --epochs 100 `
    --batch_size 4 `
    --num_workers 0 `
    --resume auto `
    --log_interval 50
```

如果想从指定checkpoint继续：

```powershell
python train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --epochs 100 `
    --batch_size 4 `
    --num_workers 0 `
    --resume D:\model_picture\checkpoints\checkpoint_epoch20.pth
```

防过拟合和稳定训练相关参数：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--patience 12` | 12 | 验证mIoU连续12轮不提升就Early Stopping |
| `--seed 42` | 42 | 固定随机种子，让训练更可复现 |
| `--class_weights auto` | auto | 从训练mask估计类别权重，缓解背景过多、缺陷过少 |
| `--class_weights none` | - | 关闭类别权重，使用普通CrossEntropy |
| `--class_weight_samples 500` | 500 | 估计类别权重时最多抽样500张mask，设0扫描全部 |

### 5.4.2 Severstal测试集训练命令

你当前用于测试的 Severstal 数据集是4类缺陷 + 背景，因此类别数应为5：

激活环境后在 VSCode PowerShell 中执行：

```powershell
python train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --epochs 2 `
    --batch_size 4 `
    --num_workers 0 `
    --log_interval 50
```

未激活环境时，也可以用完整Python路径执行：

```powershell
D:\model_picture\defect-unet\python.exe train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --epochs 2 `
    --batch_size 4 `
    --num_workers 0 `
    --log_interval 50
```

Windows 下如果 DataLoader 多进程不稳定，先用 `--num_workers 0` 跑通；正式长训练时再尝试改回 `--num_workers 4`。

### 5.5 监控训练过程

训练脚本会在终端直接打印batch级进度，例如：

```text
[训练] Epoch 1/2 | Batch 50/2199 (2.3%) | loss=0.2134 | mIoU=0.4120 | elapsed=42s | ETA=30m25s
[验证] Epoch 1/2 | Batch 50/472 (10.6%) | loss=0.1988 | mIoU=0.4301 | elapsed=18s | ETA=2m31s
```

| 字段 | 含义 | 用途 |
|---|---|---|
| `Epoch 1/2` | 当前第几轮/总轮数 | 判断整体训练跑到哪里 |
| `Batch 50/2199` | 当前epoch内第几个batch/总batch数 | 判断当前轮跑到哪里 |
| `2.3%` | 当前epoch完成百分比 | 直观看进度 |
| `loss` | 截至当前batch的平均损失 | 看训练是否下降、是否异常NaN |
| `mIoU` | 截至当前batch的平均交并比 | 看分割质量趋势 |
| `elapsed` | 当前训练/验证阶段已耗时 | 判断当前阶段已经跑了多久 |
| `ETA` | 当前训练/验证阶段预计剩余时间 | 估算本阶段还要多久 |

`--log_interval` 控制打印频率：默认50；想更频繁看进度可设为10；设为0时只打印每个训练/验证阶段的首尾。

```bash
# 另开一个终端，启动TensorBoard
E:\Miniconda\condabin\conda_hook.bat
conda activate D:\model_picture\defect-unet
tensorboard --logdir D:\model_picture\logs

# 浏览器打开 http://localhost:6006
# 查看：
#   - loss曲线（训练和验证）
#   - mIoU曲线
#   - 每个类别的IoU曲线
```

### 5.6 判断训练效果

| 指标 | 可用 | 较好 | 很好 |
|---|---|---|---|
| mIoU | > 0.45 | > 0.60 | > 0.70 |
| 划痕IoU | > 0.40 | > 0.55 | > 0.65 |
| 锈蚀IoU | > 0.50 | > 0.65 | > 0.75 |
| 裂纹IoU | > 0.30 | > 0.45 | > 0.60 |
| 压伤IoU | > 0.35 | > 0.50 | > 0.65 |
| 毛刺IoU | > 0.40 | > 0.55 | > 0.70 |

**关键看缺陷类别的IoU，不要只看mIoU（背景IoU会拉高mIoU）。**

### 5.7 训练不理想时的对策

| 问题 | 原因 | 对策 |
|---|---|---|
| 损失不下降 | lr太大或太小 | `--lr 5e-4` 或 `--lr 2e-3` |
| 损失震荡 | lr太大或batch太小 | `--lr 5e-4 --batch_size 4` |
| 训练好但验证差 | 过拟合 | 增加数据增强、增加数据量 |
| 某类缺陷IoU=0 | 该类样本太少 | 专门补充该类样本200+张 |
| 掩码边界粗糙 | 分割头不够强 | 换encoder_depth=5 |
| 掩码全黑/全0 | 掩码格式错误 | 检查掩码PNG像素值是否正确 |
| 整体IoU偏低 | 数据量不够 | 每类至少300张 |

### 5.8 分步调优策略

```
训练结果不理想？
  │
  ├── 1. 先看数据
  │     ├── 某类样本 < 150张 → 补充该类样本
  │     ├── 良品样本 < 500张 → 补充良品样本
  │     └── 掩码标注不准 → 重新标注边界
  │
  ├── 2. 再调参数
  │     ├── lr 从 1e-3 → 5e-4
  │     ├── epochs 100 → 150
  │     └── batch_size 8 → 4
  │
  ├── 3. 改损失函数
  │     ├── 缺陷面积小 → 增大Dice权重（train.py中0.5→1.0）
  │     └── 类别极度不平衡 → 加Focal Loss
  │
  ├── 4. 换模型
  │     ├── 当前环境先用 mobilenet_v2 跑通
  │     ├── 后续升级到真正 MobileNetV3-Small / MobileNetV3-Large
  │     └── smp.Unet → smp.UnetPlusPlus
  │
  └── 5. 仍不达标 → 考虑方案A（MobileNetV3+LR-ASPP）或增加注意力机制
```

### 5.9 本次环境兼容修改记录：为什么先用 MobileNetV2

| 项目 | 当前状态 |
|---|---|
| 原方案目标 | UNet + MobileNetV3-Small |
| 当前SMP版本 | `segmentation_models_pytorch==0.5.0` |
| 实测问题 | 当前SMP encoder列表不包含 `mobilenet_v3_small` / `mobilenet_v3_large` |
| 直接后果 | 继续使用 `--encoder mobilenet_v3_small` 会报 `Wrong encoder name` |
| 临时处理 | `train.py` 默认改为 `mobilenet_v2`，保证当前测试数据集能先跑通 |
| 影响 | 这不是最终部署模型选择，只是为了当前环境先验证数据、训练循环、损失函数和导出流程 |

当前环境可用的轻量encoder示例：

| encoder | 说明 | 适用场景 |
|---|---|---|
| `mobilenet_v2` | 当前默认，SMP直接支持，参数量约6.63M | 先跑通训练、验证、导出流程 |
| `efficientnet-b0` | 精度和速度较均衡 | 测试精度上限 |
| `timm-tf_efficientnet_lite0` | 面向移动端的轻量EfficientNet变体 | 后续比较移动端部署潜力 |
| `resnet18` | 经典baseline，较大 | 做精度参考，不一定适合STM32MP157部署 |

后续升级到真正 MobileNetV3 的路线：

| 路线 | 做法 | 优点 | 风险/工作量 |
|---|---|---|---|
| A. 升级/更换SMP版本 | 安装支持 MobileNetV3 encoder 的版本，重新检查 `smp.encoders.encoders.keys()` | 代码改动最少 | 版本变化可能影响依赖、权重下载、ONNX导出 |
| B. 使用 `timm` + SMP 的 timm encoder | 选择SMP支持的 `timm-*` encoder，或确认是否存在 MobileNetV3 的timm入口 | 保持SMP训练框架 | encoder名称和预训练权重需要实测确认 |
| C. 自定义MobileNetV3 encoder | 用 `torchvision.models.mobilenet_v3_small` 注册/封装为SMP encoder | 最接近原始方案 | 需要实现特征层输出、通道数、encoder_depth，工作量最大 |
| D. 改用方案A | MobileNetV3 + LR-ASPP/DeepLab类轻量分割头 | 更贴近移动端语义分割常见结构 | 训练和导出脚本要重新适配 |

升级前必须重新做三件事：

| 检查项 | 命令/动作 | 目的 |
|---|---|---|
| 确认可用encoder | `python -c "import segmentation_models_pytorch as smp; print([k for k in smp.encoders.encoders.keys() if 'mobile' in k.lower()])"` | 确认环境是否真的支持MobileNetV3 |
| 训练脚本烟雾测试 | `python train.py --data_dir ... --num_classes ... --epochs 1 --batch_size 1 --num_workers 0` | 确认能前向、反向、保存checkpoint |
| ONNX导出测试 | `python export_onnx.py --checkpoint ... --num_classes ... --encoder ...` | 确认部署链路没有断 |

---

## 6. 导出ONNX模型

训练完成后，最佳模型保存在 `./checkpoints/best_model.pth`。

注意：导出必须使用训练环境里的 Python。不要直接运行裸 `python export_onnx.py`，因为系统里的 `C:\Python313\python.exe` 没有安装 PyTorch/SMP 等训练依赖。

首次导出前确认 ONNX 包已经安装：

```bash
D:\model_picture\defect-unet\python.exe -m pip install onnx
```

```bash
# 导出ONNX
D:\model_picture\defect-unet\python.exe export_onnx.py \
    --checkpoint D:\model_picture\checkpoints\best_model.pth \
    --output D:\model_picture\checkpoints\defect_unet.onnx \
    --num_classes 5 \
    --encoder mobilenet_v2
```

导出后会生成 `defect_unet.onnx` 文件。当前 Severstal + `mobilenet_v2` 测试模型导出的文件约25MB；后续换更轻量结构或INT8量化后体积会继续下降。

导出成功时应看到类似输出：

```text
模型加载成功，训练时最佳mIoU: 0.43877813156371215
ONNX模型已导出: D:\model_picture\checkpoints\defect_unet.onnx
ONNX模型验证通过
输入: input
输出: output
文件大小: 25.3 MB
```

常见导出错误：

| 报错 | 原因 | 解决方法 |
|---|---|---|
| `No module named 'segmentation_models_pytorch'` | 用了系统Python，不是训练环境Python | 改用 `D:\model_picture\defect-unet\python.exe export_onnx.py ...` |
| `Module onnx is not installed!` | 训练环境缺少ONNX包 | 执行 `D:\model_picture\defect-unet\python.exe -m pip install onnx` |
| `类别数不一致` | `--num_classes` 和checkpoint不一致 | 按checkpoint保存的类别数填写，Severstal测试集为5 |
| `encoder不一致` | `--encoder` 和checkpoint不一致 | 按checkpoint保存的encoder填写，当前测试为 `mobilenet_v2` |

可用 Netron 可视化查看模型结构：https://netron.app

---

## 7. 电脑端UVC摄像头实时推理（上板前验证）

现在模型还没有加载到 STM32MP157 板子上时，推荐先在电脑上跑 `infer_camera_onnx.py`。这一步的价值是：先确认 **摄像头画面、ONNX模型、预处理、分割mask、OK/NG阈值** 都能跑通，再去做 NCNN/INT8 和板端移植。

### 7.1 运行前检查

| 检查项 | 命令 | 说明 |
|---|---|---|
| 激活环境 | `conda activate D:\model_picture\defect-unet` | 必须用训练环境，里面已有 OpenCV 和 onnxruntime |
| 确认摄像头编号 | `python test_camera.py` | 你的UVC摄像头编号当前是 `1` |
| 确认ONNX存在 | `Get-Item D:\model_picture\checkpoints\defect_unet.onnx` | 文件应约25MB |

### 7.2 推荐运行命令

```powershell
cd D:\model_picture
conda activate D:\model_picture\defect-unet

python infer_camera_onnx.py `
    --camera_id 1 `
    --model D:\model_picture\checkpoints\defect_unet.onnx `
    --num_classes 5
```

脚本启动后会打开实时窗口，默认显示 **原图 + 分割叠加图**。

| 按键 | 功能 |
|---|---|
| `q` 或 `ESC` | 退出实时推理 |
| `s` | 保存当前原图、叠加图、彩色mask到 `camera_infer_outputs` |

### 7.3 常用参数

| 参数 | 默认值 | 作用 | 什么时候改 |
|---|---:|---|---|
| `--camera_id` | `1` | 摄像头编号 | 打开的不是UVC时改成0或2 |
| `--model` | `checkpoints/defect_unet.onnx` | ONNX模型路径 | 换模型文件时修改 |
| `--num_classes` | `5` | 输出类别数 | Severstal测试集是5类；后续真实数据按训练类别数填写 |
| `--input_size` | `224` | 模型输入尺寸 | 必须和训练、导出一致，当前不要改 |
| `--threshold_pixels` | `80` | 非背景像素超过该值判定NG | 误报多就调大，漏检多就调小 |
| `--alpha` | `0.45` | 缺陷颜色叠加强度 | 颜色太淡就调大，遮挡原图就调小 |
| `--view` | `side_by_side` | 显示模式 | 可选 `overlay`、`side_by_side`、`mask` |
| `--provider` | `auto` | ONNX Runtime后端 | 普通环境会自动用CPU；装了GPU版ORT才会用CUDA |

示例：只看叠加图，并把NG阈值调高：

```powershell
python infer_camera_onnx.py --camera_id 1 --threshold_pixels 300 --view overlay
```

### 7.4 画面颜色和状态含义

实时窗口里的彩色区域是模型输出的分割 mask 叠加到原始摄像头画面上的结果。简单说：**没有颜色代表背景，出现颜色代表模型认为那一块不是背景，而是某个缺陷类别**。

当前 Severstal 测试模型使用 `--num_classes 5`，颜色表如下：

| 画面颜色 | 类别ID | 当前脚本显示 | 含义 |
|---|---:|---|---|
| 无颜色 | `0` | `background` | 模型认为是正常背景/非缺陷区域 |
| 红色 | `1` | `class_1` | 模型预测为第1类非背景区域 |
| 橙色 | `2` | `class_2` | 模型预测为第2类非背景区域 |
| 蓝色 | `3` | `class_3` | 模型预测为第3类非背景区域 |
| 紫色 | `4` | `class_4` | 模型预测为第4类非背景区域 |

注意：这里的 `class_1~class_4` 只是当前 Severstal 测试模型的占位类别，不等于你真实零件项目里的“划痕、锈蚀、压伤、裂纹”。后续用你自己的UVC工位数据重新训练后，才应该把这些类别名对应到真实缺陷类型。

画面左上角状态说明：

| 显示项 | 代表什么 | 怎么判断 |
|---|---|---|
| `OK` | 当前帧非背景像素数低于阈值 | 暂时认为没有明显缺陷 |
| `NG` | 当前帧非背景像素数达到或超过阈值 | 暂时认为存在缺陷或疑似缺陷 |
| `defect_px` | 当前帧中非背景像素数量 | 数值越大，彩色mask面积越大 |
| `FPS` | 实时处理帧率 | 越高越流畅 |
| `infer=xxms` | 单帧ONNX模型推理耗时 | 只统计模型推理，不含摄像头读取和显示 |
| `camera=1` | 当前使用的摄像头编号 | 你的UVC摄像头当前是1 |
| `provider=CPUExecutionProvider` | ONNX Runtime执行后端 | 当前用CPU推理；装GPU版ORT后可能显示CUDA |

例如：如果画面中螺丝孔边缘出现蓝色或紫色，不代表脚本已经知道它是“孔”或“真实缺陷”，只代表当前测试模型把那一片像素预测成了 `class_3` 或 `class_4`。由于这个模型不是用你的零件训练的，它很可能会把孔边缘阴影、反光、纹理变化误判成缺陷。

出现不同现象时可以这样理解：

| 现象 | 说明 | 处理建议 |
|---|---|---|
| 完全没有彩色区域 | 模型认为整帧都是背景 | 如果真实有缺陷但没颜色，说明漏检或模型没学会 |
| 局部出现彩色区域 | 模型认为该区域是非背景类别 | 当前测试阶段只说明链路通了，需结合真实标注模型判断 |
| 彩色区域越来越大 | `defect_px` 会升高 | 达到阈值后会从OK变成NG |
| 孔边缘、阴影、反光处有颜色 | 测试模型把结构/光照误判为缺陷 | 固定光源，后续用真实数据重新训练 |
| 一直显示NG | 阈值太小或模型误报严重 | 临时调大 `--threshold_pixels`，例如500或1000 |

临时减少误报示例：

```powershell
python infer_camera_onnx.py --camera_id 1 --threshold_pixels 500
```

### 7.5 推理预处理必须与训练一致

`infer_camera_onnx.py` 的预处理流程与 `dataset.py` 中验证集预处理一致：

| 步骤 | 电脑端脚本做法 | 为什么必须一致 |
|---|---|---|
| 通道顺序 | OpenCV BGR → RGB | 训练时 PIL 读取的是 RGB |
| 尺寸 | resize 到 `224x224` | 训练和ONNX导出固定为224 |
| 像素范围 | `uint8 0-255` → `float32 0-1` | 与 Albumentations Normalize 输入一致 |
| 归一化 | mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]` | 与 ImageNet 预训练 backbone 匹配 |
| 输出格式 | HWC → NCHW，增加 batch 维度 | ONNX输入是 `(1,3,224,224)` |

如果电脑端或后续板端推理结果明显不对，第一优先检查这里：**尺寸、RGB/BGR、mean/std、mask类别ID**。

### 7.6 当前测试模型的注意事项

| 项目 | 说明 |
|---|---|
| 当前模型来源 | Severstal钢板缺陷测试数据集，不是你的真实金属零件数据 |
| 当前类别数 | 5类：背景 + 4类Severstal缺陷 |
| 实际意义 | 用来验证训练、导出、摄像头推理链路是否跑通 |
| 不能期待 | 不能期待它准确识别你真实工位零件的划痕、锈蚀、压伤等缺陷 |
| 后续升级 | 换成你自己UVC摄像头采集、Labelme标注、重新训练出的 `best_model.pth`，再导出新的ONNX |

### 7.7 电脑端跑通后再上板

电脑端实时推理通过后，后续路线是：

```text
UVC摄像头电脑端ONNX推理通过
  ↓
确认真实数据集类别和标注质量
  ↓
重新训练真实工位模型
  ↓
导出新的 defect_unet.onnx
  ↓
ONNX → NCNN → INT8
  ↓
部署到 STM32MP157
```

---

## 8. 量化与部署到 STM32MP157

### 8.1 ONNX → NCNN 转换

在训练机器上（x86）做转换，然后把转换后的文件拷到板子上：

```bash
# 安装NCNN转换工具（Linux环境，WSL或Linux主机上执行）
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake .. && make -j4
sudo make install

# ONNX转NCNN参数文件
cd D:\model_picture
onnx2ncnn defect_unet.onnx defect_unet.param defect_unet.bin

# 优化NCNN模型（折叠常量、融合算子）
ncnnoptimize defect_unet.param defect_unet.bin defect_unet_opt.param defect_unet_opt.bin 65536
```

### 8.2 INT8 量化（NCNN方式）

```bash
# 步骤1：生成校准数据（用训练集的图片）
python -c "
import os
imgs = [f for f in os.listdir('./datasets/train/images') if f.endswith('.jpg')][:200]
with open('calib.list', 'w') as f:
    for img in imgs:
        f.write(f'./datasets/train/images/{img}\n')
"

# 步骤2：执行INT8量化
ncnn2table defect_unet_opt.param defect_unet_opt.bin calib.list defect_unet.table 224 224

# 步骤3：用INT8表重新优化模型
ncnnoptimize defect_unet_opt.param defect_unet_opt.bin defect_unet.table defect_unet_int8.param defect_unet_int8.bin 65536
```

### 8.3 在 STM32MP157 上编译 NCNN

在板子的 Linux 系统上：

```bash
# 安装编译依赖
apt-get update
apt-get install -y cmake g++ make git

# 克隆NCNN
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# 编译（开启NEON优化和INT8支持）
mkdir build && cd build
cmake -DNCNN_OPENMP=ON \
      -DNCNN_INT8=ON \
      -DNCNN_ARM=ON \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      ..
make -j2
make install
```

### 8.4 推理代码（C++，在 MP157 上编译运行）

```cpp
/*
 * defect_infer.cpp —— STM32MP157 上的缺陷分割推理
 * 编译: g++ -O2 -mfpu=neon defect_infer.cpp -I../ncnn/build/install/include \
 *       -L../ncnn/build/install/lib -lncnn -o defect_infer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

/* 类别名称 */
static const char* CLASS_NAMES[] = {
    "background", "scratch", "rust", "dent", "crack", "burr"
};
#define NUM_CLASSES 6

/*
 * 推理函数
 * 参数:
 *   net:       已加载的NCNN模型
 *   bgr_frame: OpenCV BGR格式输入图像
 *   out_mask:  输出分割掩码（每个像素=类别ID）
 */
void infer_defect(ncnn::Net& net, const cv::Mat& bgr_frame,
                  cv::Mat& out_mask)
{
    /* 1. 预处理：缩放到224x224 */
    cv::Mat resized;
    cv::resize(bgr_frame, resized, cv::Size(224, 224));

    /* 2. 转换为NCNN Mat (BGR->RGB, 归一化) */
    int w = resized.cols;
    int h = resized.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        resized.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 224, 224
    );

    /* ImageNet归一化（必须与训练时dataset.py一致） */
    const float mean[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm[3] = {1.0f / 0.229f / 255.f,
                           1.0f / 0.224f / 255.f,
                           1.0f / 0.225f / 255.f};
    in.substract_mean_normalize(mean, norm);

    /* 3. 推理 */
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

    /* 4. 后处理：取argmax得到每个像素的类别 */
    out_mask.create(224, 224, CV_8UC1);

    for (int y = 0; y < out.h; y++)
    {
        for (int x = 0; x < out.w; x++)
        {
            int best_cls = 0;
            float best_val = -1e9f;
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                float val = out.channel(c).row(y)[x];
                if (val > best_val)
                {
                    best_val = val;
                    best_cls = c;
                }
            }
            out_mask.at<uchar>(y, x) = (uchar)best_cls;
        }
    }
}

/*
 * 统计缺陷面积
 * 返回每种缺陷类别占图像的像素比例
 */
void count_defect_pixels(const cv::Mat& mask, float ratios[NUM_CLASSES])
{
    int total = mask.rows * mask.cols;
    int counts[NUM_CLASSES] = {0};

    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            int cls = mask.at<uchar>(y, x);
            if (cls >= 0 && cls < NUM_CLASSES)
            {
                counts[cls]++;
            }
        }
    }

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        ratios[i] = (float)counts[i] / (float)total;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("用法: %s <model.param> <model.bin> [image.jpg]\n", argv[0]);
        return -1;
    }

    /* 加载模型 */
    ncnn::Net net;
    net.load_param(argv[1]);
    net.load_model(argv[2]);

    /* 读取测试图片 */
    const char* image_path = (argc >= 4) ? argv[3] : "test.jpg";
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        printf("无法读取图片: %s\n", image_path);
        return -1;
    }

    /* 推理 */
    cv::Mat result_mask;
    infer_defect(net, frame, result_mask);

    /* 统计缺陷 */
    float ratios[NUM_CLASSES] = {0};
    count_defect_pixels(result_mask, ratios);

    printf("=== 检测结果 ===\n");
    int has_defect = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
    {
        if (ratios[i] > 0.005f)  /* 缺陷面积>0.5%才报告 */
        {
            printf("  %s: %.2f%%\n", CLASS_NAMES[i], ratios[i] * 100);
            has_defect = 1;
        }
    }

    if (has_defect)
    {
        printf(">>> 判定: 坏品 <<<\n");
    }
    else
    {
        printf(">>> 判定: 良品 <<<\n");
    }

    /* 保存可视化掩码（彩色） */
    cv::Mat color_mask = cv::Mat::zeros(224, 224, CV_8UC3);
    uchar colors[6][3] = {
        {0, 0, 0},       /* 背景: 黑 */
        {0, 0, 255},     /* 划痕: 红 */
        {0, 255, 0},     /* 锈蚀: 绿 */
        {255, 0, 0},     /* 压伤: 蓝 */
        {0, 255, 255},   /* 裂纹: 黄 */
        {255, 0, 255},   /* 毛刺: 紫 */
    };

    for (int y = 0; y < 224; y++)
    {
        for (int x = 0; x < 224; x++)
        {
            int cls = result_mask.at<uchar>(y, x);
            color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(
                colors[cls][0], colors[cls][1], colors[cls][2]
            );
        }
    }

    cv::imwrite("result_mask.png", color_mask);
    printf("掩码已保存: result_mask.png\n");

    return 0;
}
```

### 8.5 部署文件清单

最终需要拷贝到 STM32MP157 上的文件：

```
STM32MP157 上的目录结构:
/opt/defect_detect/
├── defect_unet_int8.param    ← NCNN INT8 模型参数（~10KB）
├── defect_unet_int8.bin      ← NCNN INT8 模型权重（~2-4MB）
├── defect_infer              ← 编译好的推理程序
└── libncnn.a                 ← NCNN静态库
```

---

## 9. 云端训练平台（没有本地GPU时）

你的本地有 RTX 4060，通常不需要云端训练。但如果需要更大显存或更多实验：

### 9.1 AutoDL（推荐国内用户）

| 步骤 | 操作 |
|---|---|
| 1 | 注册 https://www.autodl.com |
| 2 | 充值20元 |
| 3 | 创建实例 → 选 RTX 3060 或 RTX 4060（约1.5元/小时） |
| 4 | 选择镜像 → PyTorch 2.0 + Python 3.9 |
| 5 | 连接实例 → JupyterLab 或 SSH |
| 6 | 上传 `D:\model_picture` 下的代码和数据 |
| 7 | 执行 `python train.py` |
| 8 | 下载 `checkpoints/best_model.pth` |

**预估费用**：UNet+MobileNetV3-Small 训练80 epoch，约1.5-3小时，约3-5元。

### 9.2 Google Colab（免费但需要翻墙）

| 步骤 | 操作 |
|---|---|
| 1 | 打开 https://colab.research.google.com |
| 2 | 新建 Notebook |
| 3 | 运行时 → 更改运行时类型 → T4 GPU |
| 4 | 挂载 Google Drive |
| 5 | 上传代码和数据到 Drive |
| 6 | 在 Notebook 中执行训练命令 |

---

## 10. 完整操作时间线

```
第1天：环境搭建（已完成✓）
  ├── 安装 Miniconda ✓
  ├── 创建 Python 环境 ✓
  ├── 安装 PyTorch + SMP + 依赖 ✓
  ├── 创建项目目录 ✓
  └── 验证环境正常 ✓

第2-3天：数据采集
  ├── 搭建拍摄工位
  ├── 用capture.py拍摄良品 200+ 张
  ├── 用capture.py拍摄各类缺陷 150+ 张/类
  └── 总计约 1000-2000 张

第4-6天：数据标注
  ├── 安装并学习 Labelme
  ├── 标注所有图片
  ├── 运行 convert_labelme.py convert 转换
  └── 运行 convert_labelme.py split 划分

第7天：开始训练
  ├── 执行 python train.py
  ├── 监控 TensorBoard
  ├── 根据效果调整参数
  └── 可能需要补充数据

第8天：模型导出与电脑端实时推理
  ├── python export_onnx.py
  ├── python infer_camera_onnx.py --camera_id 1
  ├── 用UVC摄像头确认实时画面、mask叠加、OK/NG阈值
  └── 根据误报/漏检决定是否补真实工位数据

第9天：量化与转换
  ├── onnx2ncnn 转换
  ├── INT8 量化
  └── 测试量化后精度

第10天：部署到板子
  ├── 在 MP157 上编译 NCNN
  ├── 拷贝模型文件
  ├── 编译推理代码
  └── 端到端测试
```

---

## 11. 常见问题排查

### 11.1 环境问题

| 问题 | 原因 | 解决 |
|---|---|---|
| VSCode PowerShell 中 `conda` 命令找不到 | Conda PowerShell hook未加载 | 先执行 `& "E:\Miniconda\shell\condabin\conda-hook.ps1"`，再执行 `conda activate D:\model_picture\defect-unet` |
| CMD 中 `conda` 命令找不到 | Conda CMD hook未加载 | 先执行 `E:\Miniconda\condabin\conda_hook.bat` |
| `nvidia-smi` 无输出 | 未安装驱动 | 安装 NVIDIA 驱动 |
| `torch.cuda.is_available()` 返回 False | CUDA版本不匹配 | 确认安装了cu124版PyTorch |
| `No module named smp` | 未激活conda环境 | 先 `conda activate D:\model_picture\defect-unet` |

### 11.2 训练问题

| 问题 | 原因 | 解决 |
|---|---|---|
| CUDA Out of Memory | batch_size太大 | 改 `--batch_size 4` 或2 |
| Loss=NaN | lr太大 | 改 `--lr 1e-4` |
| 某类IoU始终为0 | 样本太少或掩码错误 | 检查掩码像素值，补充数据 |
| 掩码全黑 | 掩码格式错误 | 用图像查看器检查掩码PNG |
| 训练速度慢 | num_workers=0 | 改 `--num_workers 4` |
| BrokenPipeError | Windows多进程问题 | 改 `--num_workers 0` |
| 图片和掩码数量不一致 | 转换时部分失败 | 检查raw_images中的JSON是否完整 |

### 11.3 电脑端UVC推理问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `No module named 'onnxruntime'` | 没有激活训练环境或环境缺包 | 先 `conda activate D:\model_picture\defect-unet`；仍失败则 `python -m pip install onnxruntime` |
| 打开的不是UVC摄像头 | `--camera_id` 编号不对 | 先运行 `python test_camera.py`，当前UVC使用 `--camera_id 1` |
| 摄像头无法打开 | 被其他软件占用或驱动未释放 | 关闭相机、微信、浏览器等占用摄像头的软件，拔插UVC后重试 |
| 实时画面卡顿 | CPU推理较慢或窗口显示压力大 | 改 `--view overlay`，或后续安装 `onnxruntime-gpu` |
| mask大面积乱报 | 当前模型是Severstal测试集，不是你的真实零件数据 | 用真实UVC工位图片重新标注训练，再导出新的ONNX |
| 推理结果和验证集差很多 | 预处理不一致或摄像头光照差异大 | 检查RGB/BGR、224尺寸、mean/std；固定光源并补充相同工位数据 |
| 一直显示NG | `--threshold_pixels` 太小或测试模型误报 | 先把阈值调大，例如 `--threshold_pixels 500`，真实模型训练好后再定阈值 |

### 11.4 部署问题

| 问题 | 原因 | 解决 |
|---|---|---|
| ONNX导出失败 | opset版本不支持 | 改 `opset_version=12` |
| NCNN推理结果不对 | 预处理不一致 | 检查归一化参数和通道顺序 |
| INT8量化后精度大降 | 校准数据不够 | 增加200+张校准图 |
| 板上推理OOM | 模型太大 | 确认使用INT8版本 |
