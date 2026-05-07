"""
MobileNetV3-Small 分类模型 ONNX 导出脚本

用途:
  将 train_classify.py 训练好的 .pth 导出为 ONNX 格式
  供后续 INT8 量化和 STM32MP157 部署使用

用法:
  使用默认参数:
    D:\model_picture\defect-unet\python.exe export_classify_onnx.py

  指定路径:
    D:\model_picture\defect-unet\python.exe export_classify_onnx.py --checkpoint ./checkpoints_classify/best_model.pth --output ./checkpoints_classify/defect_classifier.onnx

导出规格:
  输入:  (1, 3, 224, 224)  float32 RGB 图片
  输出:  (1, 2)            float32 每类概率 (good, bad)

前置条件:
  必须先完成分类模型训练，checkpoints_classify/ 下有 best_model.pth
"""

import argparse
import os

import torch
from torchvision import models


def build_model(num_classes=2):
    """
    构建 MobileNetV3-Small 分类模型（与 train_classify.py 一致）

    参数:
        num_classes (int): 输出类别数，默认 2 (good/bad)

    返回:
        model (nn.Module): 构建好的模型
    """
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def export_onnx(checkpoint_path, onnx_path, num_classes):
    """
    导出 ONNX 模型

    参数:
        checkpoint_path (str): 训练好的 .pth 文件路径
        onnx_path (str): 导出的 ONNX 文件路径
        num_classes (int): 类别数

    返回:
        None

    主要流程:
      1. 构建模型结构
      2. 加载训练好的权重
      3. 设置模型为 eval 模式
      4. 构造 dummy input 并导出 ONNX
    """
    print(f"加载模型: {checkpoint_path}")

    # 构建模型
    model = build_model(num_classes=num_classes)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 兼容不同保存格式
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  从 checkpoint 加载 (epoch={checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"  直接加载 state_dict")

    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 构造 dummy input: (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出 ONNX
    print(f"\n导出 ONNX: {onnx_path}")

    # 确保输出目录存在
    out_dir = os.path.dirname(onnx_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # 打印导出结果
    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  导出成功!")
    print(f"  文件大小: {file_size:.2f} MB")
    print(f"  输入规格: (batch, 3, 224, 224)")
    print(f"  输出规格: (batch, {num_classes})")
    print(f"\n下一步:")
    print(f"  1. 用 Netron 可视化: https://netron.app")
    print(f"  2. INT8 量化后部署到 STM32MP157")
    print(f"  3. 预估量化后大小: ~1~2 MB")


def main():
    """
    主入口: 解析命令行参数并执行导出
    """
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small 分类模型 ONNX 导出"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints_classify/best_model.pth",
        help="训练好的模型权重路径"
    )
    parser.add_argument(
        "--output", type=str, default="./checkpoints_classify/defect_classifier.onnx",
        help="导出的 ONNX 文件路径"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="类别数，默认 2 (good/bad)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"找不到模型权重: {args.checkpoint}\n"
            f"请先运行 train_classify.py 训练分类模型。"
        )

    export_onnx(args.checkpoint, args.output, args.num_classes)


if __name__ == "__main__":
    main()
