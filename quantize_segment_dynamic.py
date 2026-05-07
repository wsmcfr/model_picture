"""
UNet 分割模型 INT8 动态量化脚本

用途:
  将 export_onnx.py 导出的 FP32 分割 ONNX 模型做动态量化
  不需要校准数据集，一步完成量化

  MobileNetV2 版本: ~25 MB -> ~6~8 MB
  MobileNetV3 版本: ~14 MB -> ~3~5 MB

与静态量化的对比:
  +------------------+----------------------+----------------------+
  | 特性             | 动态量化 (本脚本)     | 静态量化             |
  +------------------+----------------------+----------------------+
  | 校准数据         | 不需要               | 需要 50~200 张图片   |
  | 权重             | INT8                 | INT8                 |
  | 激活值           | 推理时动态量化        | 校准阶段预先量化      |
  | 推理速度         | 中等                 | 最快（ARM 最优）      |
  | 精度损失         | 通常略大             | 通常更小             |
  | 推荐场景         | 快速测试             | 生产部署             |
  +------------------+----------------------+----------------------+

用法:
  基础动态量化:
    D:\model_picture\defect-unet\python.exe quantize_segment_dynamic.py \
        --onnx_input ./checkpoints/defect_unet.onnx \
        --onnx_output ./checkpoints/defect_unet_dynamic_int8.onnx

  量化 MobileNetV3 UNet:
    D:\model_picture\defect-unet\python.exe quantize_segment_dynamic.py \
        --onnx_input ./checkpoints_mobilenetv3/defect_unet_mobilenetv3.onnx \
        --onnx_output ./checkpoints_mobilenetv3/defect_unet_mobilenetv3_dynamic_int8.onnx
"""

import argparse
import os

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)


def main():
    """
    主入口：解析参数并执行 UNet 分割模型动态量化

    动态量化流程:
      1. 加载 FP32 ONNX 模型
      2. 将 Conv / MatMul 等算子的权重转为 INT8
      3. 插入 DynamicQuantizeLinear 节点（推理时动态量化激活值）
      4. 保存量化后的 ONNX 模型

    不需要校准数据，直接对权重做量化即可。
    """
    parser = argparse.ArgumentParser(
        description="UNet 分割模型 INT8 动态量化"
    )

    # ---- 输入输出路径 ----
    parser.add_argument(
        "--onnx_input", type=str,
        default="./checkpoints/defect_unet.onnx",
        help="输入 FP32 ONNX 模型路径"
    )
    parser.add_argument(
        "--onnx_output", type=str,
        default="./checkpoints/defect_unet_dynamic_int8.onnx",
        help="输出动态 INT8 ONNX 模型路径"
    )

    # ---- 量化参数 ----
    parser.add_argument(
        "--weight_type", type=str, default="qint8",
        choices=["qint8", "quint8"],
        help="权重量化类型，默认 qint8（推荐 ARM CPU）"
    )
    parser.add_argument(
        "--optimize_model", type=int, default=1, choices=[0, 1],
        help="是否先做图优化再量化（1=是，0=否），默认 1"
    )

    args = parser.parse_args()

    # 检查输入模型
    if not os.path.exists(args.onnx_input):
        raise FileNotFoundError(
            f"找不到输入 ONNX 模型: {args.onnx_input}\n"
            f"请先运行 export_onnx.py 导出 FP32 分割模型。"
        )

    # 确保输出目录存在
    out_dir = os.path.dirname(args.onnx_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 解析权重量化类型
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    print(f"[动态量化配置]")
    print(f"  输入模型:     {args.onnx_input}")
    print(f"  输出模型:     {args.onnx_output}")
    print(f"  权重量化类型: {args.weight_type}")
    print(f"  图优化:       {'开启' if args.optimize_model else '关闭'}")
    print(f"\n[开始量化] 动态量化不需要校准数据，直接处理权重...")

    # 执行动态量化
    quantize_dynamic(
        model_input=args.onnx_input,
        model_output=args.onnx_output,
        weight_type=weight_type,
        optimize_model=bool(args.optimize_model),
    )

    # 打印结果
    in_size = os.path.getsize(args.onnx_input) / (1024 * 1024)
    out_size = os.path.getsize(args.onnx_output) / (1024 * 1024)
    ratio = in_size / out_size if out_size > 0 else 0

    print(f"\n[量化完成]")
    print(f"  FP32 模型: {args.onnx_input}")
    print(f"    大小: {in_size:.2f} MB")
    print(f"  INT8 动态量化模型: {args.onnx_output}")
    print(f"    大小: {out_size:.2f} MB")
    print(f"  压缩比: {ratio:.1f}x")
    print(f"\n[说明]")
    print(f"  动态量化模型适合快速测试，无需校准数据。")
    print(f"  但推理时激活值需要实时量化，速度略逊于静态量化。")
    print(f"  如需最佳性能，请使用 quantize_segment_int8.py 做静态量化。")
    print(f"\n[下一步]")
    print(f"  1. 测试量化模型: python infer_camera_onnx.py --model {args.onnx_output}")
    print(f"  2. 对比 FP32 vs INT8 分割效果")
    print(f"  3. 若精度损失大，换用静态量化并增加校准图片数量")


if __name__ == "__main__":
    main()
