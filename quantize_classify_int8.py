"""
MobileNetV3-Small 分类模型 INT8 静态量化脚本

用途:
  将 export_classify_onnx.py 导出的 FP32 ONNX 模型量化为 INT8
  显著减小模型体积 (~5MB -> ~1.5MB) 并加速 ARM CPU 推理

量化原理:
  静态量化（Static Quantization）在模型转换阶段预先收集激活值的分布范围，
  将 FP32 权重和激活都映射到 INT8 整数范围 [-128, 127]。
  与动态量化相比，静态量化在推理时不需要实时统计范围，延迟更低，
  非常适合 STM32MP157 这类无 NPU 的嵌入式 CPU。

前置条件:
  1. 已导出 FP32 ONNX 模型 (defect_classifier.onnx)
  2. 有少量（50~200张）代表性的校准图片，覆盖 good/bad 两类
  3. 校准图片的预处理方式必须与推理时完全一致

用法:
  快速量化（使用当前目录下图片作为校准集）:
    D:\model_picture\defect-unet\python.exe quantize_classify_int8.py \
        --onnx_input ./checkpoints_classify/defect_classifier.onnx \
        --onnx_output ./checkpoints_classify/defect_classifier_int8.onnx \
        --calib_dir ./datasets_classify/val \
        --num_calib 100

  只使用 good 图做校准:
    D:\model_picture\defect-unet\python.exe quantize_classify_int8.py \
        --calib_dir ./datasets_classify/train/good \
        --num_calib 50

  指定自定义预处理尺寸（如果你训练时不是 224x224）:
    D:\model_picture\defect-unet\python.exe quantize_classify_int8.py \
        --input_size 224
"""

import argparse
import os
import glob
import random

import cv2
import numpy as np

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)


# ============================================================
# 全局常量：必须与训练 / 推理时完全一致
# ============================================================

# ImageNet 归一化参数，与 infer_classify.py / train_classify.py 保持一致
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_quant(image_bgr, input_size=224):
    """
    量化校准用的预处理函数

    必须与模型实际推理时的预处理完全一致！
    任何差异都会导致量化后的模型精度下降。

    参数:
        image_bgr (np.ndarray): OpenCV 读取的 BGR 图片，uint8
        input_size (int): 输入边长，默认 224

    返回:
        input_tensor (np.ndarray): 形状 (1, 3, input_size, input_size)，float32
    """
    # 直接 Resize 到正方形（与训练时的 Resize((224,224)) 一致）
    resized = cv2.resize(image_bgr, (input_size, input_size))

    # BGR -> RGB
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 归一化到 [0,1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # ImageNet 标准化
    image_norm = (image_float - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    image_chw = np.transpose(image_norm, (2, 0, 1))

    # 增加 batch 维度 -> (1, 3, H, W)
    input_tensor = np.expand_dims(image_chw, axis=0).astype(np.float32)

    return input_tensor


# ============================================================
# 校准数据读取器（ONNX Runtime 量化 API 要求）
# ============================================================

class ClassifyCalibrationDataReader(CalibrationDataReader):
    """
    分类模型校准数据读取器

    ONNX Runtime 的 quantize_static 需要一个 CalibrationDataReader 对象，
    用来逐批次提供校准数据。框架会在内部用这些数据跑一遍模型，
    收集每层激活值的 min/max 范围，作为量化参数。

    参数:
        image_paths (list[str]): 校准图片路径列表
        input_size (int): 模型输入尺寸
        batch_size (int): 每批次喂给模型的图片数（通常为1）

    说明:
        校准图片数量建议 50~200 张，覆盖所有类别和典型场景。
        太少：统计范围不准，量化误差大
        太多：校准耗时长，收益递减
    """

    def __init__(self, image_paths, input_size=224, batch_size=1):
        """
        初始化校准数据读取器

        参数:
            image_paths (list[str]): 图片路径列表
            input_size (int): 输入边长
            batch_size (int): 批次大小
        """
        self.image_paths = image_paths
        self.input_size = input_size
        self.batch_size = batch_size
        self.index = 0

    def get_next(self):
        """
        获取下一批校准数据

        ONNX Runtime 量化框架会循环调用此方法，直到返回 None 表示结束。

        返回:
            dict or None:
                成功时返回 {"input": numpy_array}，失败/结束返回 None
                numpy_array 形状必须是 (batch, 3, H, W)
        """
        if self.index >= len(self.image_paths):
            return None

        # 收集当前批次的图片
        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        batch_tensors = []

        for path in batch_paths:
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                print(f"[警告] 无法读取校准图片，跳过: {path}")
                continue
            tensor = preprocess_for_quant(image_bgr, self.input_size)
            batch_tensors.append(tensor)

        if not batch_tensors:
            return None

        # 合并批次: [(1,3,H,W), ...] -> (N, 3, H, W)
        batch_data = np.concatenate(batch_tensors, axis=0)

        self.index += len(batch_paths)

        # ONNX Runtime 要求返回字典，key 是输入节点名称
        # 默认输入名是 "input"（与 export_classify_onnx.py 中设置的 input_names 一致）
        return {"input": batch_data}


def collect_calibration_images(calib_dir, num_calib=100):
    """
    从目录中收集校准图片路径

    参数:
        calib_dir (str): 校准数据根目录
        num_calib (int): 需要收集的最大图片数量

    返回:
        image_paths (list[str]): 图片路径列表

    说明:
        会在 calib_dir 及其子目录中递归搜索 .jpg/.jpeg/.png/.bmp 文件。
        如果图片多于 num_calib，随机采样；如果少于，则全部使用。
    """
    # 支持的图片格式
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []

    # 递归搜索所有子目录（支持 ImageFolder 结构）
    for ext in exts:
        pattern = os.path.join(calib_dir, "**", ext)
        image_paths.extend(glob.glob(pattern, recursive=True))

    if not image_paths:
        raise FileNotFoundError(
            f"在 {calib_dir} 及其子目录中未找到任何图片 (.jpg/.jpeg/.png/.bmp)\n"
            f"请准备校准数据集，建议从训练集/验证集中抽取 50~200 张。"
        )

    # 随机打乱后采样，保证覆盖均匀
    random.shuffle(image_paths)

    if len(image_paths) > num_calib:
        image_paths = image_paths[:num_calib]

    print(f"[校准数据] 从 {calib_dir} 收集到 {len(image_paths)} 张图片用于量化校准")
    return image_paths


def main():
    """
    主入口：解析参数、收集校准数据、执行静态量化
    """
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small 分类模型 INT8 静态量化"
    )

    # ---- 输入输出路径 ----
    parser.add_argument(
        "--onnx_input", type=str,
        default="./checkpoints_classify/defect_classifier.onnx",
        help="输入 FP32 ONNX 模型路径"
    )
    parser.add_argument(
        "--onnx_output", type=str,
        default="./checkpoints_classify/defect_classifier_int8.onnx",
        help="输出 INT8 ONNX 模型路径"
    )

    # ---- 校准数据参数 ----
    parser.add_argument(
        "--calib_dir", type=str,
        default="./datasets_classify/val",
        help="校准图片目录，支持子目录递归搜索"
    )
    parser.add_argument(
        "--num_calib", type=int, default=100,
        help="校准图片数量，建议 50~200，默认 100"
    )

    # ---- 预处理参数 ----
    parser.add_argument(
        "--input_size", type=int, default=224,
        help="模型输入尺寸，必须与训练时一致，默认 224"
    )

    # ---- 量化参数 ----
    parser.add_argument(
        "--quant_format", type=str, default="qdq",
        choices=["qdq", "qoperator"],
        help="量化格式: qdq=QDQ格式(推荐ARM), qoperator=QOperator格式"
    )
    parser.add_argument(
        "--activation_type", type=str, default="qint8",
        choices=["qint8", "quint8"],
        help="激活值量化类型，ARM CPU 推荐 qint8"
    )
    parser.add_argument(
        "--weight_type", type=str, default="qint8",
        choices=["qint8", "quint8"],
        help="权重量化类型，ARM CPU 推荐 qint8"
    )

    args = parser.parse_args()

    # 检查输入模型是否存在
    if not os.path.exists(args.onnx_input):
        raise FileNotFoundError(
            f"找不到输入 ONNX 模型: {args.onnx_input}\n"
            f"请先运行 export_classify_onnx.py 导出 FP32 模型。"
        )

    # 确保输出目录存在
    out_dir = os.path.dirname(args.onnx_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 收集校准图片
    calib_images = collect_calibration_images(args.calib_dir, args.num_calib)

    # 创建校准数据读取器
    calib_reader = ClassifyCalibrationDataReader(
        image_paths=calib_images,
        input_size=args.input_size,
        batch_size=1,
    )

    # 解析量化格式和类型
    quant_format = QuantFormat.QDQ if args.quant_format == "qdq" else QuantFormat.QOperator
    activation_type = QuantType.QInt8 if args.activation_type == "qint8" else QuantType.QUInt8
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    print(f"\n[量化配置]")
    print(f"  输入模型:  {args.onnx_input}")
    print(f"  输出模型:  {args.onnx_output}")
    print(f"  量化格式:  {args.quant_format.upper()}")
    print(f"  激活类型:  {args.activation_type}")
    print(f"  权重类型:  {args.weight_type}")
    print(f"  校准图片:  {len(calib_images)} 张")
    print(f"  输入尺寸:  {args.input_size}x{args.input_size}")
    print(f"\n[开始量化] 正在收集激活范围统计，请稍候...")

    # 执行静态量化
    # optimize_model=True 会先对模型做图优化，再量化，通常效果更好
    quantize_static(
        model_input=args.onnx_input,
        model_output=args.onnx_output,
        calibration_data_reader=calib_reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        optimize_model=True,
    )

    # 打印结果
    in_size = os.path.getsize(args.onnx_input) / (1024 * 1024)
    out_size = os.path.getsize(args.onnx_output) / (1024 * 1024)
    ratio = in_size / out_size if out_size > 0 else 0

    print(f"\n[量化完成]")
    print(f"  FP32 模型: {args.onnx_input}")
    print(f"    大小: {in_size:.2f} MB")
    print(f"  INT8 模型: {args.onnx_output}")
    print(f"    大小: {out_size:.2f} MB")
    print(f"  压缩比: {ratio:.1f}x")
    print(f"\n[下一步]")
    print(f"  1. 用 infer_classify.py 测试 INT8 模型精度:")
    print(f"     python infer_classify.py --model {args.onnx_output} --mode batch --input <测试集>")
    print(f"  2. 对比 FP32 vs INT8 精度损失，通常 < 1%")
    print(f"  3. 将 INT8 模型复制到 STM32MP157 部署")


if __name__ == "__main__":
    main()
