# model_picture

UNet + MobileNet 轻量缺陷分割项目，用于在 Windows 电脑上训练、导出 ONNX，并在上板前通过 UVC 摄像头做实时推理验证。

## 当前状态

| 项目 | 状态 |
|---|---|
| 训练脚本 | `train.py`，支持进度显示、续训、best/last checkpoint、Early Stopping |
| ONNX导出 | `export_onnx.py` |
| 电脑端UVC实时推理 | `infer_camera_onnx.py`，默认摄像头编号为 `1` |
| 当前测试模型 | `checkpoints/defect_unet.onnx`，基于 Severstal 测试数据集 |
| 详细手册 | `方案B_UNet_MobileNetV3缺陷分割训练部署完全手册.md` |

## 注意

当前仓库没有上传本地虚拟环境、训练数据集、TensorBoard日志和 `.pth` checkpoint，因为这些文件体积很大，不适合直接放进 GitHub。

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

## 电脑端UVC实时推理

```powershell
cd D:\model_picture
conda activate D:\model_picture\defect-unet

python infer_camera_onnx.py `
    --camera_id 1 `
    --model D:\model_picture\checkpoints\defect_unet.onnx `
    --num_classes 5
```

| 按键 | 功能 |
|---|---|
| `q` 或 `ESC` | 退出 |
| `s` | 保存当前原图、叠加图和彩色mask |

颜色含义：

| 颜色 | 类别ID | 当前含义 |
|---|---:|---|
| 无颜色 | 0 | 背景 |
| 红色 | 1 | class_1 |
| 橙色 | 2 | class_2 |
| 蓝色 | 3 | class_3 |
| 紫色 | 4 | class_4 |

## 训练

```powershell
python train.py `
    --data_dir D:\model_picture\datasets_severstal `
    --num_classes 5 `
    --epochs 100 `
    --batch_size 4 `
    --num_workers 0 `
    --log_interval 50
```

中断后续训：

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

## 导出ONNX

```powershell
python export_onnx.py `
    --checkpoint D:\model_picture\checkpoints\best_model.pth `
    --output D:\model_picture\checkpoints\defect_unet.onnx `
    --num_classes 5 `
    --encoder mobilenet_v2
```

## 测试

```powershell
python -m unittest tests.test_infer_camera_onnx -v
python -m py_compile train.py dataset.py export_onnx.py infer_camera_onnx.py
```

## 文档

完整训练、续训、导出、电脑端推理、后续上板部署路线请看：

```text
方案B_UNet_MobileNetV3缺陷分割训练部署完全手册.md
```
