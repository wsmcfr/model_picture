"""
将训练好的PyTorch模型(.pth)导出为ONNX格式(.onnx)

用途:
  ONNX是跨框架的模型交换格式，PyTorch训练的模型需要转成ONNX
  才能在NCNN/TFLite等推理框架上使用

  完整部署链路:
  PyTorch训练 → .pth → ONNX → NCNN → INT8量化 → STM32MP157推理

  本脚本负责第二步: .pth → ONNX

用法:
  使用默认参数:
    python export_onnx.py

  指定自定义路径:
    python export_onnx.py --checkpoint ./checkpoints/best_model.pth --output ./checkpoints/defect_unet.onnx

  如果改变了类别数或backbone:
    python export_onnx.py --num_classes 5 --encoder mobilenet_v2

前置条件:
  必须先完成训练，checkpoints/下有best_model.pth文件

导出后的文件:
  defect_unet.onnx (~5-8MB)
  可用Netron可视化查看模型结构: https://netron.app
"""

import argparse
import os

# 关闭HuggingFace Hub在Windows下关于缓存软链接的提示，避免导出日志被无关警告刷屏。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import segmentation_models_pytorch as smp


RECOMMENDED_ENCODERS = [
    "mobilenet_v2",
    "tu-mobilenetv3_small_100.lamb_in1k",
    "tu-tf_mobilenetv3_small_100.in1k",
    "efficientnet-b0",
    "timm-tf_efficientnet_lite0",
    "mobileone_s0",
    "resnet18",
]


def build_encoder_help_examples():
    """
    生成导出脚本错误提示中使用的encoder示例列表。

    主要流程：
      1. 普通encoder从SMP静态列表中确认是否存在。
      2. `tu-*`动态timm encoder不在静态列表中，但SMP 0.5.0可以创建，因此单独保留。

    返回值：
        list[str]：当前项目推荐使用的encoder名称。
    """
    supported_encoders = set(smp.encoders.encoders.keys())
    examples = []
    for name in RECOMMENDED_ENCODERS:
        if name.startswith("tu-") or name in supported_encoders:
            examples.append(name)
    return examples


def validate_encoder_name(encoder):
    """
    校验导出时传入的encoder名称。

    参数：
        encoder (str)：命令行传入的backbone名称，必须与checkpoint保存的一致。

    返回值：
        None。校验不通过时抛出ValueError。

    说明：
      MobileNetV3对比实验推荐 `tu-mobilenetv3_small_100.lamb_in1k`。
      这个名称通过SMP的timm通用入口创建，不会出现在静态encoder列表里。
    """
    supported_encoders = set(smp.encoders.encoders.keys())
    if encoder in supported_encoders or encoder.startswith("tu-"):
        return

    examples = ", ".join(build_encoder_help_examples())
    raise ValueError(
        f"当前SMP环境不支持encoder={encoder!r}。"
        f"可用轻量encoder示例: {examples}。"
        "导出时的encoder必须与训练保存checkpoint时一致。"
    )


def export_onnx(checkpoint_path, onnx_path, num_classes, encoder):
    """
    导出ONNX模型

    步骤:
      1. 创建与训练时结构完全一致的模型
      2. 加载训练好的权重(.pth文件)
      3. 设置为评估模式(model.eval)
      4. 用torch.onnx.export将模型导出为ONNX格式

    参数:
        checkpoint_path (str):
            训练好的.pth权重文件路径
            通常在 ./checkpoints/best_model.pth
            文件内包含: model_state_dict, optimizer_state_dict, epoch, best_miou等字段

        onnx_path (str):
            输出ONNX文件路径
            通常保存到 ./checkpoints/defect_unet.onnx
            文件大小约5-8MB（取决于backbone）

        num_classes (int):
            分割类别数（含背景），必须与训练时一致
            训练时5类，这里也必须是5
            如果不一致，权重加载会报错

        encoder (str):
            backbone名称，必须与训练时一致
            例如训练时mobilenet_v2，这里也必须是mobilenet_v2
            如果不一致，模型结构不同，权重无法加载

    为什么创建模型时要设encoder_weights=None?
      encoder_weights="imagenet" 会从网络下载预训练权重
      但我们已经有本地训练好的权重了，不需要预训练权重
      设None避免不必要的下载，后面用load_state_dict加载本地权重

    为什么用torch.randn作为dummy_input?
      ONNX导出需要跟踪一次前向传播来确定模型结构
      dummy_input的形状必须与实际推理时的输入一致: (1, 3, 224, 224)
        1 = batch_size（固定为1，部署时单张推理）
        3 = RGB三通道
        224 = 高度（与训练时Resize一致）
        224 = 宽度
      数值内容无所谓，只需要形状正确

    opset_version=11 的选择:
      ONNX用opset版本号定义支持的算子集
      11: 兼容性好，NCNN/onnxruntime都支持
      12+: 支持更多新算子，但部分推理框架可能不支持
      如果NCNN转换报错"unsupported op"，可以试opset_version=9或12

    dynamic_axes=None 的含义:
      固定输入尺寸，不支持动态batch/分辨率
      优点: 模型结构确定，量化部署更简单
      缺点: 只能处理224x224的输入
      对于STM32MP157部署，固定尺寸是合理的（推理时预处理resize到224）
    """
    # 1. 创建模型（结构与训练时完全一致）
    # 先校验encoder名称，避免SMP抛出很长的KeyError后不容易看清真正问题。
    validate_encoder_name(encoder)

    model = smp.Unet(
        encoder_name=encoder,           # backbone，必须与训练时一致
        encoder_weights=None,           # 不加载预训练，用本地训练好的权重
        in_channels=3,                  # RGB三通道输入
        classes=num_classes,            # 输出类别数，必须与训练时一致
    )

    # 2. 加载训练好的权重
    # map_location="cpu": 将权重加载到CPU，不依赖GPU
    # 这样即使在没有GPU的机器上也能导出ONNX
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # checkpoint里如果保存了训练配置，就提前检查，避免load_state_dict才报结构不匹配。
    ckpt_num_classes = checkpoint.get('num_classes')
    ckpt_encoder = checkpoint.get('encoder')
    if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
        raise ValueError(
            f"类别数不一致: checkpoint是{ckpt_num_classes}类，"
            f"当前导出参数是{num_classes}类。请把--num_classes改成{ckpt_num_classes}。"
        )
    if ckpt_encoder is not None and ckpt_encoder != encoder:
        raise ValueError(
            f"encoder不一致: checkpoint使用{ckpt_encoder!r}，"
            f"当前导出参数是{encoder!r}。请把--encoder改成{ckpt_encoder}。"
        )

    # load_state_dict: 将权重字典加载到模型中
    # 严格模式(strict=True): 权重字典的key必须与模型的参数名完全匹配
    # 如果报错"Missing key"或"Unexpected key"，说明模型结构不一致
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. 设置为评估模式
    # eval()关闭Dropout和BatchNorm的训练行为
    # 导出时必须用eval()，否则BatchNorm的运行统计量不对
    model.eval()

    print(f"模型加载成功，训练时最佳mIoU: {checkpoint.get('best_miou', 'N/A')}")

    # 4. 创建虚拟输入
    # 形状必须与推理时一致: (batch=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 5. 导出ONNX
    # torch.onnx.export 会执行一次前向传播，记录所有计算操作
    # 然后将这些操作转换为ONNX格式的计算图
    torch.onnx.export(
        model,                      # 要导出的模型
        dummy_input,                # 虚拟输入（用于跟踪前向传播）
        onnx_path,                  # 输出ONNX文件路径
        opset_version=11,           # ONNX算子版本，11兼容性最好
        input_names=["input"],      # 输入节点名称，NCNN推理时用这个名字
        output_names=["output"],    # 输出节点名称，NCNN推理时用这个名字
        dynamic_axes=None,          # 固定输入尺寸，不支持动态形状
    )

    print(f"ONNX模型已导出: {onnx_path}")

    # 6. 验证ONNX模型
    # onnx.checker.check_model 检查ONNX文件的结构是否合法
    # 会验证: 算子是否支持、输入输出是否匹配、图结构是否完整
    # 如果验证失败，说明导出过程有问题，NCNN转换也会失败
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")

    # 打印模型输入输出信息
    print(f"输入: {onnx_model.graph.input[0].name}")
    print(f"输出: {onnx_model.graph.output[0].name}")

    # 计算文件大小
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"文件大小: {file_size:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将训练好的PyTorch模型导出为ONNX格式，用于NCNN转换和部署"
    )

    # --checkpoint: 训练好的权重文件路径
    # 由 train.py 训练完成后自动保存
    # best_model.pth: 验证mIoU最高时的模型（推荐使用）
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/best_model.pth",
        help="训练好的.pth权重文件路径（默认./checkpoints/best_model.pth）"
    )

    # --output: 输出ONNX文件路径
    # 后续NCNN转换用这个文件作为输入
    parser.add_argument(
        "--output", type=str,
        default="./checkpoints/defect_unet.onnx",
        help="输出ONNX文件路径（默认./checkpoints/defect_unet.onnx）"
    )

    # --num_classes: 类别数
    # 必须与训练时一致，否则权重加载报错
    parser.add_argument(
        "--num_classes", type=int, default=6,
        help="类别数（含背景），必须与训练时一致；Severstal测试集通常用5"
    )

    # --encoder: backbone名称
    # 必须与训练时一致，否则模型结构不同
    parser.add_argument(
        "--encoder", type=str, default="mobilenet_v2",
        help="backbone名称，必须与训练时一致（默认mobilenet_v2；V3对比可用tu-mobilenetv3_small_100.lamb_in1k）"
    )

    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.num_classes, args.encoder)
