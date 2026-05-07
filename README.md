# model_picture

UNet + MobileNet 轻量缺陷分割项目，以及 MobileNetV3-Small 良坏分类模型，用于在 Windows 电脑上训练、导出 ONNX、INT8 量化，并在上板前通过 UVC 摄像头做实时推理验证。

## 当前状态

| 项目 | 状态 |
|---|---|
| 分割训练脚本 | `train.py`，支持进度显示、续训、best/last checkpoint、Early Stopping |
| 分割 ONNX 导出 | `export_onnx.py` |
| 分割 UVC 实时推理 | `infer_camera_onnx.py`，默认摄像头编号为 `1` |
| 分割量化（动态/静态） | `quantize_segment_dynamic.py` / `quantize_segment_int8.py` |
| 分类训练脚本 | `train_classify.py`，MobileNetV3-Small 二分类（good/bad） |
| 分类 ONNX 导出 | `export_classify_onnx.py` |
| 分类推理 | `infer_classify.py`，支持单图/批量/摄像头三种模式 |
| 分类量化（动态/静态） | `quantize_classify_dynamic.py` / `quantize_classify_int8.py` |
| 当前分割模型 | `checkpoints/defect_unet.onnx`，MobileNetV2 骨干，25.27 MB |
| MobileNetV3 分割模型 | `checkpoints_mobilenetv3/defect_unet_mobilenetv3.onnx`，13.70 MB |
| 分类模型 | 待训练（需先准备 `datasets_classify/` 数据） |
| 详细手册 | `方案B_UNet_MobileNetV3缺陷分割训练部署完全手册.md` |

## 注意

当前仓库没有上传本地虚拟环境、训练数据集、TensorBoard 日志和 `.pth` checkpoint，因为这些文件体积很大，不适合直接放进 GitHub。

已保留 `checkpoints/defect_unet.onnx`，方便直接测试电脑端 UVC 摄像头推理链路。这个 ONNX 模型只用于验证流程，不代表已经能准确识别真实零件缺陷。

## 环境

推荐 Python 3.9。

```powershell
cd D:\model_picture
conda create -p D:\model_picture\defect-unet python=3.9 -y
conda activate D:\model_picture\defect-unet
python -m pip install -r requirements.txt
```

如果使用 CUDA 版本 PyTorch，建议按 PyTorch 官网与你的显卡驱动匹配安装。当前本机环境是 PyTorch `2.6.0+cu124`。

---

## 分割模型（UNet + MobileNet）

### 电脑端 UVC 实时推理

```powershell
python infer_camera_onnx.py `
    --camera_id 1 `
    --model D:\model_picture\checkpoints\defect_unet.onnx `
    --num_classes 5
```

| 按键 | 功能 |
|---|---|
| `q` 或 `ESC` | 退出 |
| `s` | 保存当前原图、叠加图和彩色 mask |

### 训练

默认 MobileNetV2：

```powershell
python train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --epochs 100 `
    --batch_size 4 `
    --num_workers 0 `
    --log_interval 50
```

MobileNetV3-Small 对比实验：

```powershell
python train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --encoder tu-mobilenetv3_small_100.lamb_in1k `
    --checkpoint_dir D:\model_picture\checkpoints_mobilenetv3 `
    --log_dir D:\model_picture\logs_mobilenetv3 `
    --epochs 100 `
    --batch_size 4 `
    --num_workers 0 `
    --log_interval 50
```

断点续训：

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

### 导出 ONNX

```powershell
# MobileNetV2
python export_onnx.py `
    --checkpoint D:\model_picture\checkpoints\best_model.pth `
    --output D:\model_picture\checkpoints\defect_unet.onnx `
    --num_classes 5 `
    --encoder mobilenet_v2

# MobileNetV3-Small
python export_onnx.py `
    --checkpoint D:\model_picture\checkpoints_mobilenetv3\best_model.pth `
    --output D:\model_picture\checkpoints_mobilenetv3\defect_unet_mobilenetv3.onnx `
    --num_classes 5 `
    --encoder tu-mobilenetv3_small_100.lamb_in1k
```

### 量化

```powershell
# 动态量化（不需要校准数据）
python quantize_segment_dynamic.py `
    --onnx_input ./checkpoints/defect_unet.onnx `
    --onnx_output ./checkpoints/defect_unet_dynamic_int8.onnx

# 静态量化（需要校准图片，精度更好）
python quantize_segment_int8.py `
    --onnx_input ./checkpoints/defect_unet.onnx `
    --onnx_output ./checkpoints/defect_unet_int8.onnx `
    --calib_dir ./datasets/val/images `
    --num_calib 100
```

---

## 分类模型（MobileNetV3-Small）

用于零件良坏二分类（good/bad），作为 STM32MP157 第一阶段主链路模型。

### 数据准备

按以下结构组织分类数据集：

```
datasets_classify/
  train/
    good/        ← 良品图片
    bad/         ← 缺陷图片
  val/
    good/
    bad/
```

### 训练

```powershell
python train_classify.py `
    --data_dir ./datasets_classify `
    --epochs 100 `
    --batch_size 16
```

### 导出 ONNX

```powershell
python export_classify_onnx.py `
    --checkpoint ./checkpoints_classify/best_model.pth `
    --output ./checkpoints_classify/defect_classifier.onnx
```

### 推理

```powershell
# 单张图片
python infer_classify.py --mode image --input ./test.jpg

# 摄像头实时
python infer_classify.py --mode camera --camera_id 0

# 批量目录
python infer_classify.py --mode batch --input ./test_images/
```

### 量化

```powershell
# 动态量化（不需要校准数据）
python quantize_classify_dynamic.py `
    --onnx_input ./checkpoints_classify/defect_classifier.onnx `
    --onnx_output ./checkpoints_classify/defect_classifier_dynamic_int8.onnx

# 静态量化（推荐，精度更好）
python quantize_classify_int8.py `
    --onnx_input ./checkpoints_classify/defect_classifier.onnx `
    --onnx_output ./checkpoints_classify/defect_classifier_int8.onnx `
    --calib_dir ./datasets_classify/val `
    --num_calib 100
```

---

## 测试

```powershell
python -m unittest tests.test_infer_camera_onnx -v
python -m py_compile train.py dataset.py export_onnx.py infer_camera_onnx.py
python -m py_compile train_classify.py export_classify_onnx.py infer_classify.py
```

## 文档

| 文档 | 用途 |
|---|---|
| `操作手册.md` | 日常训练操作速查 |
| `方案B_UNet_MobileNetV3缺陷分割训练部署完全手册.md` | 从零到部署的完整流程 |
| `项目全景说明.md` | 项目全局视角、模型定位、下一步计划 |
