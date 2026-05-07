"""
UNet 分割模型 INT8 静态量化脚本

用途:
  将 export_onnx.py 导出的 FP32 分割 ONNX 模型量化为 INT8
  显著减小模型体积并加速 ARM CPU 推理

  MobileNetV2 版本: ~25 MB -> ~6~8 MB
  MobileNetV3 版本: ~14 MB -> ~3~5 MB

与分类量化的核心区别:
  1. 预处理: 分割使用 LongestMaxSize + PadIfNeeded（保持比例），
     分类使用直接 Resize(224,224)
  2. 输出: 分割输出 (1, num_classes, 224, 224) 像素级概率图
  3. 校准数据来源: 分割数据集的 val/images/ 目录

用法:
  量化 MobileNetV2 UNet:
    D:\model_picture\defect-unet\python.exe quantize_segment_int8.py \
        --onnx_input ./checkpoints/defect_unet.onnx \
        --onnx_output ./checkpoints/defect_unet_int8.onnx \
        --calib_dir ./datasets/val/images \
        --num_calib 100

  量化 MobileNetV3 UNet:
    D:\model_picture\defect-unet\python.exe quantize_segment_int8.py \
        --onnx_input ./checkpoints_mobilenetv3/defect_unet_mobilenetv3.onnx \
        --onnx_output ./checkpoints_mobilenetv3/defect_unet_mobilenetv3_int8.onnx \
        --calib_dir ./datasets/val/images \
        --num_calib 100
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_segment(image_bgr, input_size=224):
    """
    分割模型校准预处理

    处理流程与 dataset.py 的 get_validation_augmentation() + get_preprocessing() 完全一致：
      1. LongestMaxSize: 保持长宽比，长边缩放到 input_size
      2. PadIfNeeded: 黑色填充到 input_size x input_size，居中放置
      3. BGR -> RGB 通道转换
      4. ImageNet 均值标准差归一化
      5. HWC -> CHW 维度转换，增加 batch 维度

    参数:
        image_bgr (np.ndarray): OpenCV 读取的 BGR 图片，uint8，形状 (H, W, 3)
        input_size (int): 目标输入边长，默认 224（与训练/导出一致）

    返回:
        input_tensor (np.ndarray): 形状 (1, 3, input_size, input_size)，float32
    """
    h, w = image_bgr.shape[:2]

    # 步骤1: 保持比例缩放（LongestMaxSize）
    # 计算缩放比例，使长边等于 input_size，短边按比例缩放
    # 例如 640x480 -> 长边640缩放为224，比例0.35，短边480->168
    scale = input_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 步骤2: 居中黑边填充到正方形（PadIfNeeded）
    # 计算四边填充量，使图片居中，总尺寸为 input_size x input_size
    # 例如 224x168 -> 上下各填充 28px，左右填充 0px
    pad_top = (input_size - new_h) // 2
    pad_bottom = input_size - new_h - pad_top
    pad_left = (input_size - new_w) // 2
    pad_right = input_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # 步骤3: BGR -> RGB
    # OpenCV 读取的是 BGR，训练时 PIL 读取的是 RGB，必须转换
    image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # 步骤4: ImageNet 归一化
    # pixel = (pixel / 255 - mean) / std
    image_float = image_rgb.astype(np.float32) / 255.0
    image_norm = (image_float - IMAGENET_MEAN) / IMAGENET_STD

    # 步骤5: HWC -> CHW，增加 batch 维度
    # (224, 224, 3) -> (3, 224, 224) -> (1, 3, 224, 224)
    image_chw = np.transpose(image_norm, (2, 0, 1))
    input_tensor = np.expand_dims(image_chw, axis=0).astype(np.float32)

    return input_tensor


# ============================================================
# 校准数据读取器
# ============================================================

class SegmentCalibrationDataReader(CalibrationDataReader):
    """
    分割模型校准数据读取器

    ONNX Runtime 的 quantize_static 通过此类逐批次获取校准数据，
    在前向传播过程中收集每一层激活值的 min/max 范围，
    最终生成静态量化参数（scale 和 zero_point）。

    参数:
        image_paths (list[str]): 校准图片路径列表
        input_size (int): 模型输入边长
        batch_size (int): 每批次图片数量（通常为 1，避免显存/内存不足）
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

        ONNX Runtime 量化框架循环调用此方法直到返回 None。

        返回:
            dict or None:
                成功时返回 {"input": numpy_array}，numpy_array 形状 (N, 3, H, W)
                数据耗尽时返回 None
        """
        if self.index >= len(self.image_paths):
            return None

        # 收集当前批次的图片路径
        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        batch_tensors = []

        for path in batch_paths:
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                print(f"[警告] 无法读取校准图片，跳过: {path}")
                continue
            tensor = preprocess_segment(image_bgr, self.input_size)
            batch_tensors.append(tensor)

        if not batch_tensors:
            return None

        # 合并批次: [(1,3,H,W), ...] -> (N, 3, H, W)
        batch_data = np.concatenate(batch_tensors, axis=0)
        self.index += len(batch_paths)

        # ONNX 输入节点名称为 "input"（与 export_onnx.py 中 input_names 一致）
        return {"input": batch_data}


# ============================================================
# 辅助函数
# ============================================================

def collect_calibration_images(image_dir, num_calib=100):
    """
    从目录中递归收集校准图片

    参数:
        image_dir (str): 图片根目录（支持子目录递归搜索）
        num_calib (int): 最大收集数量，默认 100

    返回:
        image_paths (list[str]): 图片路径列表

    说明:
        支持 .jpg/.jpeg/.png/.bmp 格式。
        如果图片数量超过 num_calib，随机采样保证分布均匀。
        如果不足，则全部使用。
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []

    for ext in exts:
        pattern = os.path.join(image_dir, "**", ext)
        image_paths.extend(glob.glob(pattern, recursive=True))

    if not image_paths:
        raise FileNotFoundError(
            f"在 {image_dir} 及其子目录中未找到任何图片\n"
            f"请准备校准数据集，建议从分割验证集或测试集中抽取 50~200 张。"
        )

    random.shuffle(image_paths)

    if len(image_paths) > num_calib:
        image_paths = image_paths[:num_calib]

    print(f"[校准数据] 从 {image_dir} 收集到 {len(image_paths)} 张图片用于量化校准")
    return image_paths


def main():
    """
    主入口：解析参数、收集校准数据、执行 UNet 分割模型静态量化
    """
    parser = argparse.ArgumentParser(
        description="UNet 分割模型 INT8 静态量化"
    )

    # ---- 输入输出路径 ----
    parser.add_argument(
        "--onnx_input", type=str,
        default="./checkpoints/defect_unet.onnx",
        help="输入 FP32 ONNX 模型路径"
    )
    parser.add_argument(
        "--onnx_output", type=str,
        default="./checkpoints/defect_unet_int8.onnx",
        help="输出 INT8 ONNX 模型路径"
    )

    # ---- 校准数据参数 ----
    parser.add_argument(
        "--calib_dir", type=str,
        default="./datasets/val/images",
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

    # 收集校准图片
    calib_images = collect_calibration_images(args.calib_dir, args.num_calib)

    # 创建校准数据读取器
    calib_reader = SegmentCalibrationDataReader(
        image_paths=calib_images,
        input_size=args.input_size,
        batch_size=1,
    )

    # 解析量化配置
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
    quantize_static(
        model_input=args.onnx_input,
        model_output=args.onnx_output,
        calibration_data_reader=calib_reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        optimize_model=True,
    )

    # 打印量化结果
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
    print(f"  1. 用 infer_camera_onnx.py 测试 INT8 模型:")
    print(f"     python infer_camera_onnx.py --model {args.onnx_output} --camera_id 0")
    print(f"  2. 对比 FP32 vs INT8 分割效果，检查边缘精度损失")
    print(f"  3. 将 INT8 模型复制到 STM32MP157 部署")


if __name__ == "__main__":
    main()
