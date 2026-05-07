"""
MobileNetV3-Small 分类模型 INT8 动态量化脚本

用途:
  将 export_classify_onnx.py 导出的 FP32 ONNX 模型做动态量化
  动态量化只对模型权重做 INT8 转换，激活值在推理时实时量化，
  因此不需要校准数据集，实现简单、一步到位。

与静态量化的区别:
  +------------------+----------------------+----------------------+
  | 特性             | 动态量化 (本脚本)     | 静态量化 (quantize_classify_int8.py) |
  +------------------+----------------------+----------------------+
  | 校准数据         | 不需要               | 需要 50~200 张图片   |
  | 权重             | INT8                 | INT8                 |
  | 激活值           | 推理时动态量化        | 校准阶段预先量化      |
  | 推理速度         | 中等（有实时转置开销）| 最快（无运行时开销）  |
  | 模型体积         | 约缩小 3~4x          | 约缩小 3~4x          |
  | 精度损失         | 通常略大于静态量化    | 通常更小             |
  | 推荐场景         | 快速测试、无校准数据   | 生产部署、ARM CPU     |
  +------------------+----------------------+----------------------+

前置条件:
  仅需 FP32 ONNX 模型，不需要任何校准图片

用法:
  基础用法（默认 QOperator 格式）:
    D:\model_picture\defect-unet\python.exe quantize_classify_dynamic.py \
        --onnx_input ./checkpoints_classify/defect_classifier.onnx \
        --onnx_output ./checkpoints_classify/defect_classifier_dynamic_int8.onnx

  指定量化类型:
    D:\model_picture\defect-unet\python.exe quantize_classify_dynamic.py \
        --weight_type qint8
"""

import argparse
import os

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)


def main():
    """
    主入口：解析参数并执行动态量化

    动态量化的流程非常简单：
      1. 加载 FP32 ONNX 模型
      2. 将可量化的算子（Conv, MatMul, Linear 等）的权重转为 INT8
      3. 插入 DynamicQuantizeLinear 节点，在推理时把激活值动态转 INT8
      4. 保存量化后的 ONNX 模型
    """
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small 分类模型 INT8 动态量化"
    )

    # ---- 输入输出路径 ----
    parser.add_argument(
        "--onnx_input", type=str,
        default="./checkpoints_classify/defect_classifier.onnx",
        help="输入 FP32 ONNX 模型路径"
    )
    parser.add_argument(
        "--onnx_output", type=str,
        default="./checkpoints_classify/defect_classifier_dynamic_int8.onnx",
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
        help="是否先做图优化再量化（1=是，0=否），默认1"
    )

    args = parser.parse_args()

    # 检查输入模型
    if not os.path.exists(args.onnx_input):
        raise FileNotFoundError(
            f"找不到输入 ONNX 模型: {args.onnx_input}\n"
            f"请先运行 export_classify_onnx.py 导出 FP32 模型。"
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
    # per_channel=True 按通道量化权重，通常精度更好，但某些后端不支持
    # reduce_range=False 使用完整的 [-128, 127] 范围
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
    print(f"  动态量化模型适合快速测试，无需校准数据即可生成。")
    print(f"  但推理时激活值需要实时量化，速度略逊于静态量化。")
    print(f"  如需最佳性能，请使用 quantize_classify_int8.py 做静态量化。")
    print(f"\n[下一步]")
    print(f"  1. 测试量化模型: python infer_classify.py --model {args.onnx_output}")
    print(f"  2. 若精度损失大，换用静态量化并增加校准图片数量")


if __name__ == "__main__":
    main()
