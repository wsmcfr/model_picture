"""
UNet + 轻量级Encoder 金属零件缺陷分割训练脚本

项目背景:
  目标是训练一个像素级缺陷分割模型，部署到STM32MP157（双核Cortex-A7 @ 800MHz，无GPU）
  检测金属零件（垫圈、冲压件）表面的5类缺陷：划痕、锈蚀、压伤、裂纹、毛刺

模型结构:
  Encoder: 默认使用MobileNetV2（当前SMP环境支持的轻量backbone，ImageNet预训练）
  Decoder: UNet标准解码器（带跳跃连接，保留缺陷边界细节）
  输入: (1, 3, 224, 224) RGB图片
  输出: (1, 6, 224, 224) 6类分割概率图

训练策略:
  损失函数: CrossEntropy + 0.5*Dice 联合损失
    - CE负责像素级分类
    - Dice负责区域级重叠度，对缺陷小目标更敏感
  优化器: AdamW（比Adam更好的权重衰减）
  学习率: 余弦退火调度（从初始lr慢慢降到接近0）
  数据增强: 翻转/旋转/噪声/亮度/遮挡等（见dataset.py）

用法:
  标准训练:
    python train.py --data_dir ./datasets --epochs 100 --batch_size 8

  显存不够时:
    python train.py --batch_size 4

  自定义参数:
    python train.py --data_dir ./datasets --epochs 150 --lr 5e-4 --batch_size 4

  监控训练（另开终端）:
    tensorboard --logdir ./logs
    浏览器打开 http://localhost:6006
"""

import os
import argparse
import time
import random

# 关闭HuggingFace Hub在Windows下关于缓存软链接的提示，避免训练日志被无关警告刷屏。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from dataset import (
    DefectDataset,
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)

# 类别名称列表（必须与dataset.py中的CLASSES一致）
# 索引0=background, 1=scratch, 2=rust, 3=dent, 4=crack, 5=burr
CLASS_NAMES = ["background", "scratch", "rust", "dent", "crack", "burr"]


# 推荐encoder名称。默认保留MobileNetV2，MobileNetV3作为对比实验通过SMP的timm通用入口启用。
# 注意：SMP 0.5.0的静态encoder列表不会列出tu-*名称，但get_encoder支持这类动态timm encoder。
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
    生成错误提示和命令行help里使用的encoder示例列表。

    主要流程：
      1. 读取SMP静态encoder列表，筛掉当前环境完全不存在的静态encoder。
      2. 额外保留tu-*动态timm encoder，因为这类名称不在静态列表里，但SMP可以创建。

    返回值：
        list[str]：当前项目推荐尝试的encoder名称。
    """
    supported_encoders = set(smp.encoders.encoders.keys())
    examples = []
    for name in RECOMMENDED_ENCODERS:
        if name.startswith("tu-") or name in supported_encoders:
            examples.append(name)
    return examples


def validate_encoder_name(encoder):
    """
    提前校验encoder名称是否可能被SMP创建。

    参数：
        encoder (str)：命令行传入的backbone名称。

    返回值：
        None。校验不通过时抛出ValueError。

    关键说明：
      - 普通encoder必须出现在smp.encoders.encoders静态列表里。
      - `tu-*`是SMP 0.5.0支持的timm通用入口，不在静态列表里，因此单独放行。
      - 真正创建模型时如果timm模型名不存在，SMP仍会抛出更具体的错误。
    """
    supported_encoders = set(smp.encoders.encoders.keys())
    if encoder in supported_encoders or encoder.startswith("tu-"):
        return

    examples = ", ".join(build_encoder_help_examples())
    raise ValueError(
        f"当前SMP环境不支持encoder={encoder!r}。"
        f"可先用这些轻量encoder测试: {examples}。"
        "MobileNetV3对比推荐使用 --encoder tu-mobilenetv3_small_100.lamb_in1k。"
    )


def set_random_seed(seed):
    """
    固定训练过程中的主要随机源，减少每次运行之间的随机波动。

    主要流程:
      1. 固定Python内置random模块，影响普通随机逻辑。
      2. 固定NumPy随机数，影响部分数据处理/增强逻辑。
      3. 固定PyTorch CPU/GPU随机数，影响模型初始化和CUDA算子。
      4. 设置cuDNN benchmark=False，避免为追求速度而选择不稳定算法。

    参数:
        seed (int): 随机种子；相同代码、数据和环境下，种子一致可提高结果可复现性。

    返回:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 这两个设置会略微牺牲一点速度，但更适合调试和对比实验。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """
    为DataLoader的每个worker设置独立但可复现的随机种子。

    主要流程:
      1. PyTorch会为每个worker生成initial_seed。
      2. 将该seed映射到NumPy/Python random可接受的范围。
      3. 让多进程数据增强在固定总seed时仍然可复现。

    参数:
        worker_id (int): DataLoader传入的worker编号，本函数不直接使用但保留签名。

    返回:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_resume_path(resume, checkpoint_dir):
    """
    根据命令行resume参数解析实际要加载的checkpoint路径。

    主要流程:
      1. 空字符串表示不续训，返回None。
      2. "auto"表示自动加载checkpoint_dir下的last_checkpoint.pth。
      3. 其他值按用户给出的路径直接使用。

    参数:
        resume (str): 命令行--resume传入的值。
        checkpoint_dir (str): checkpoint保存目录。

    返回:
        str or None: 需要加载的checkpoint路径；None表示从头训练。
    """
    if not resume:
        return None
    if resume.lower() == "auto":
        return os.path.join(checkpoint_dir, "last_checkpoint.pth")
    return resume


def validate_checkpoint_config(checkpoint, num_classes, encoder):
    """
    检查checkpoint中的模型结构配置是否与当前命令行参数一致。

    主要流程:
      1. 如果旧checkpoint没有保存配置，则跳过对应检查，保持兼容。
      2. 如果保存了num_classes或encoder，则必须与当前参数一致。
      3. 不一致时提前报清楚错误，避免load_state_dict输出大量难懂信息。

    参数:
        checkpoint (dict): torch.load读出的checkpoint字典。
        num_classes (int): 当前命令行指定的类别数。
        encoder (str): 当前命令行指定的encoder名称。

    返回:
        None
    """
    ckpt_num_classes = checkpoint.get("num_classes")
    ckpt_encoder = checkpoint.get("encoder")

    if ckpt_num_classes is not None and ckpt_num_classes != num_classes:
        raise ValueError(
            f"checkpoint类别数是{ckpt_num_classes}，当前--num_classes是{num_classes}，"
            "两者不一致，不能直接续训。"
        )
    if ckpt_encoder is not None and ckpt_encoder != encoder:
        raise ValueError(
            f"checkpoint encoder是{ckpt_encoder!r}，当前--encoder是{encoder!r}，"
            "两者不一致，不能直接续训。"
        )


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_miou,
                    num_classes, encoder, epochs_without_improve):
    """
    保存完整训练状态，既能用于部署选择，也能用于断点续训。

    主要流程:
      1. 保存模型权重，这是推理和继续训练的核心。
      2. 保存优化器和学习率调度器状态，续训时学习率/动量不会断档。
      3. 保存best_miou和连续未提升轮数，续训后Early Stopping还能接着判断。
      4. 保存num_classes和encoder，防止用错误结构加载checkpoint。

    参数:
        path (str): checkpoint输出路径。
        epoch (int): 当前epoch索引，从0开始保存，恢复时会从epoch+1继续。
        model (nn.Module): 当前训练模型。
        optimizer (torch.optim.Optimizer): 当前优化器。
        scheduler: 当前学习率调度器。
        best_miou (float): 历史最佳验证mIoU。
        num_classes (int): 类别数。
        encoder (str): encoder名称。
        epochs_without_improve (int): 验证集连续未提升轮数。

    返回:
        None
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_miou": best_miou,
        "num_classes": num_classes,
        "encoder": encoder,
        "epochs_without_improve": epochs_without_improve,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device, num_classes, encoder):
    """
    加载checkpoint并恢复训练状态。

    主要流程:
      1. 从磁盘读取checkpoint，如果文件不存在则直接报错。
      2. 校验num_classes/encoder，保证模型结构匹配。
      3. 恢复模型、优化器、调度器状态。
      4. 计算下一轮应从哪个epoch开始。

    参数:
        path (str): checkpoint路径。
        model (nn.Module): 已创建好的模型实例。
        optimizer (torch.optim.Optimizer): 已创建好的优化器实例。
        scheduler: 已创建好的学习率调度器实例。
        device (torch.device): 当前训练设备。
        num_classes (int): 当前类别数。
        encoder (str): 当前encoder名称。

    返回:
        tuple[int, float, int]: (start_epoch, best_miou, epochs_without_improve)。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到要续训的checkpoint: {path}")

    checkpoint = torch.load(path, map_location=device)
    validate_checkpoint_config(checkpoint, num_classes, encoder)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 兼容旧checkpoint：如果没有scheduler状态，就让调度器从当前命令行重新开始。
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        print("提示: checkpoint中没有scheduler_state_dict，学习率调度器将按当前参数重新开始。")

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_miou = checkpoint.get("best_miou", 0.0)
    epochs_without_improve = checkpoint.get("epochs_without_improve", 0)

    return start_epoch, best_miou, epochs_without_improve


def compute_class_weights_from_masks(masks_dir, num_classes, max_files=500):
    """
    从训练集mask像素分布估计CrossEntropy类别权重。

    主要流程:
      1. 抽样读取训练mask，统计每个类别的像素数量。
      2. 使用反频率思想给少数类更高权重，缓解背景像素主导损失的问题。
      3. 将权重归一到均值约为1，并限制最大值，避免极少数类权重过大导致训练不稳定。

    参数:
        masks_dir (str): 训练集mask目录。
        num_classes (int): 类别数。
        max_files (int): 最多抽样多少张mask；None或<=0表示扫描全部。

    返回:
        torch.Tensor: shape=(num_classes,) 的float32类别权重。
    """
    mask_names = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith((".png", ".bmp"))
    ])
    if max_files and max_files > 0:
        mask_names = mask_names[:max_files]

    pixel_counts = np.zeros(num_classes, dtype=np.float64)
    for mask_name in mask_names:
        mask_path = os.path.join(masks_dir, mask_name)
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        for cls in range(num_classes):
            pixel_counts[cls] += np.sum(mask == cls)

    # 为完全没抽到的类别加一个平滑值，防止除零；该类仍会得到较高但有限的权重。
    smoothed_counts = pixel_counts + 1.0
    frequencies = smoothed_counts / smoothed_counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 5.0)

    return torch.tensor(weights, dtype=torch.float32)


def build_class_names(num_classes):
    """
    根据实际训练类别数生成日志中使用的类别名称列表。

    主要流程:
      1. 如果num_classes不超过内置金属缺陷类别数，直接截取对应名称。
      2. 如果测试其他数据集且类别更多，用class_N补齐，避免日志循环越界。

    参数:
        num_classes (int): 训练时传入的类别总数，必须包含背景类。

    返回:
        list[str]: 长度等于num_classes的类别名列表，用于打印和TensorBoard记录。
    """
    if num_classes <= len(CLASS_NAMES):
        return CLASS_NAMES[:num_classes]

    # 额外类别没有业务名称时，用通用名称占位，保证日志数量与模型输出通道一致。
    extra_names = [f"class_{idx}" for idx in range(len(CLASS_NAMES), num_classes)]
    return CLASS_NAMES + extra_names


def format_seconds(seconds):
    """
    将秒数格式化为更适合训练日志阅读的时间字符串。

    主要流程:
      1. 小于1分钟时显示为Xs，方便看单个epoch内的短耗时。
      2. 小于1小时时显示为XmYs，方便估算剩余训练时间。
      3. 超过1小时时显示为XhYmZs，长时间训练时更直观。

    参数:
        seconds (float): 需要格式化的秒数，可以是已耗时或预计剩余时间。

    返回:
        str: 形如"12s"、"3m20s"或"1h02m03s"的时间文本。
    """
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def should_log_progress(batch_idx, total_batches, log_interval):
    """
    判断当前batch是否需要打印进度。

    主要流程:
      1. 第1个batch必定打印，确认训练已经真正开始。
      2. 最后1个batch必定打印，确认本阶段已经结束。
      3. 中间每隔log_interval个batch打印一次，避免终端刷屏。
      4. log_interval<=0时只打印首尾，便于需要极简日志时使用。

    参数:
        batch_idx (int): 当前batch序号，从1开始。
        total_batches (int): 当前epoch总batch数。
        log_interval (int): 打印间隔，单位是batch。

    返回:
        bool: True表示需要打印当前进度，False表示跳过。
    """
    if batch_idx == 1 or batch_idx == total_batches:
        return True
    if log_interval <= 0:
        return False
    return batch_idx % log_interval == 0


def format_progress_line(stage, epoch, total_epochs, batch_idx, total_batches,
                         avg_loss, avg_miou, elapsed_seconds, eta_seconds):
    """
    生成训练/验证阶段的单行进度文本。

    主要流程:
      1. 计算当前batch百分比，告诉用户已经跑到当前epoch的多少。
      2. 展示当前阶段的平均loss和平均mIoU，方便判断训练是否异常。
      3. 展示已耗时和ETA，便于估算本轮还要多久。

    参数:
        stage (str): 当前阶段名称，例如"训练"或"验证"。
        epoch (int): 当前epoch序号，从1开始。
        total_epochs (int): 总epoch数。
        batch_idx (int): 当前batch序号，从1开始。
        total_batches (int): 当前epoch总batch数。
        avg_loss (float): 截至当前batch的平均损失。
        avg_miou (float): 截至当前batch的平均mIoU。
        elapsed_seconds (float): 当前阶段已经耗费的秒数。
        eta_seconds (float): 当前阶段预计剩余秒数。

    返回:
        str: 可直接print的进度行。
    """
    percent = batch_idx / total_batches * 100 if total_batches > 0 else 0.0
    return (
        f"[{stage}] Epoch {epoch}/{total_epochs} | "
        f"Batch {batch_idx}/{total_batches} ({percent:.1f}%) | "
        f"loss={avg_loss:.4f} | mIoU={avg_miou:.4f} | "
        f"elapsed={format_seconds(elapsed_seconds)} | ETA={format_seconds(eta_seconds)}"
    )


def summarize_ious(all_ious, num_classes):
    """
    汇总一个阶段内累计收集到的IoU列表。

    主要流程:
      1. 对每个类别分别求平均IoU，没有有效样本的类别记为0。
      2. mIoU只排除真正不存在的NaN类别，保留预测失败得到的0分。
      3. 训练和验证共用这一段逻辑，避免两处统计方式不一致。

    参数:
        all_ious (dict[int, list[float]]): 每个类别已经收集到的IoU列表。
        num_classes (int): 类别总数。

    返回:
        tuple[float, dict[int, float]]: (mIoU, 每个类别平均IoU字典)。
    """
    avg_ious = {}
    for cls in range(num_classes):
        if len(all_ious[cls]) > 0:
            avg_ious[cls] = sum(all_ious[cls]) / len(all_ious[cls])
        else:
            avg_ious[cls] = 0.0

    # 只跳过NaN代表的“不存在类别”；IoU=0说明该类存在但预测失败，必须计入均值。
    valid_ious = [v for values in all_ious.values() for v in values]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    return miou, avg_ious


def log_stage_progress(stage, epoch, total_epochs, batch_idx, total_batches,
                       total_loss, all_ious, num_classes, stage_start_time,
                       log_interval):
    """
    按固定间隔打印训练/验证阶段进度。

    主要流程:
      1. 使用should_log_progress判断当前batch是否需要输出。
      2. 用当前累计loss和IoU计算阶段内滚动平均值。
      3. 根据已耗时和batch进度估算ETA。
      4. flush=True强制立即刷新终端，避免长时间看不到输出。

    参数:
        stage (str): 当前阶段名称，例如"训练"或"验证"。
        epoch (int): 当前epoch序号，从1开始。
        total_epochs (int): 总epoch数。
        batch_idx (int): 当前batch序号，从1开始。
        total_batches (int): 当前epoch总batch数。
        total_loss (float): 当前阶段累计loss。
        all_ious (dict[int, list[float]]): 当前阶段累计IoU。
        num_classes (int): 类别总数。
        stage_start_time (float): 当前阶段开始时间，来自time.time()。
        log_interval (int): 打印间隔，单位是batch。
    """
    if not should_log_progress(batch_idx, total_batches, log_interval):
        return

    elapsed_seconds = time.time() - stage_start_time
    avg_loss = total_loss / batch_idx
    avg_miou, _ = summarize_ious(all_ious, num_classes)
    seconds_per_batch = elapsed_seconds / batch_idx if batch_idx > 0 else 0.0
    eta_seconds = seconds_per_batch * max(total_batches - batch_idx, 0)

    print(
        format_progress_line(
            stage=stage,
            epoch=epoch,
            total_epochs=total_epochs,
            batch_idx=batch_idx,
            total_batches=total_batches,
            avg_loss=avg_loss,
            avg_miou=avg_miou,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
        ),
        flush=True,
    )


def parse_args():
    """
    解析命令行参数

    每个参数的含义和推荐值见下方help说明
    """
    parser = argparse.ArgumentParser(
        description="UNet + 轻量级Encoder 金属零件缺陷分割训练"
    )

    # --data_dir: 数据集根目录
    # 目录下必须包含 train/images, train/masks, val/images, val/masks 子目录
    # 由 convert_labelme.py split 命令自动生成
    parser.add_argument(
        "--data_dir", type=str, default="./datasets",
        help="数据集根目录（需含train/val子目录，由convert_labelme.py split生成）"
    )

    # --num_classes: 分割类别数（含背景）
    # 背景1类 + 5种缺陷 = 6类
    # 如果只检测部分缺陷类型，需要对应减少
    # 例如只检测划痕和锈蚀: num_classes=3（背景+划痕+锈蚀）
    parser.add_argument(
        "--num_classes", type=int, default=6,
        help="分割类别数（含背景），默认6=背景+5种缺陷"
    )

    # --batch_size: 每次送入模型的图片数量
    # 越大训练越稳定，但显存占用越多
    # RTX 4060 8GB: batch_size=8 大约占4-5GB显存，没问题
    # 显存不够时改4或2，训练效果可能略差但不会跑不了
    # 注意：batch_size太小（1-2）可能导致BatchNorm统计不准确
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="批大小（RTX 4060 8GB可用8，显存不够改4或2）"
    )

    # --epochs: 训练总轮数
    # 每一轮(epoch)会遍历整个训练集一次
    # 100轮通常是起步值，看效果后再决定是否继续
    # UNet+MobileNetV3一般50-80轮收敛，但100轮保险
    # 可以先跑100轮，如果还在改善就加到150
    # 注意：配合--resume续训时，epochs表示“训练到第几轮结束”，不是额外再跑多少轮
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练总轮数（默认100；续训时表示训练到第几轮结束）"
    )

    # --lr: 初始学习率
    # 学习率控制参数更新的步长
    # 太大: 损失震荡不下降，甚至NaN
    # 太小: 收敛太慢，可能陷入局部最优
    # 1e-3 (0.001) 是UNet常用的初始学习率，配合AdamW效果不错
    # 如果损失震荡: 降到5e-4
    # 如果损失下降太慢: 可以试2e-3
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="初始学习率（默认1e-3，损失震荡降到5e-4，下降太慢试2e-3）"
    )

    # --encoder: backbone编码器名称
    # 默认保留mobilenet_v2，保证当前已经跑通的训练/导出/推理链路不受影响。
    # MobileNetV3对比实验推荐使用:
    #   tu-mobilenetv3_small_100.lamb_in1k
    # 这个名称来自SMP 0.5.0的timm通用入口，虽然不在静态encoder列表里，但可以正常创建UNet。
    # 更换backbone后checkpoint不能混用，必须用新的checkpoint_dir和output路径保存。
    parser.add_argument(
        "--encoder", type=str, default="mobilenet_v2",
        help="backbone名称（默认mobilenet_v2；V3对比可用tu-mobilenetv3_small_100.lamb_in1k）"
    )

    # --checkpoint_dir: 模型权重保存目录
    # best_model.pth: 验证mIoU最高时保存的模型（最常用）
    # checkpoint_epochXX.pth: 每20轮保存的检查点（用于恢复训练）
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="模型权重保存目录（best_model.pth + 每20轮检查点）"
    )

    # --log_dir: TensorBoard日志目录
    # 训练过程中自动记录loss/mIoU/各类IoU曲线
    # 用 tensorboard --logdir ./logs 查看可视化
    parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="TensorBoard日志目录（用tensorboard --logdir ./logs查看）"
    )

    # --num_workers: DataLoader的数据加载线程数
    # 多线程并行加载图片，避免GPU等数据
    # Windows下设4通常没问题
    # 如果报错"BrokenPipeError"，改成0（单线程，慢但稳定）
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="数据加载线程数（默认4，报BrokenPipeError改0）"
    )

    # --log_interval: 训练/验证进度打印间隔
    # 每隔多少个batch打印一次当前epoch进度、平均loss、平均mIoU、耗时和ETA
    # 默认50适合大多数训练；想看更密集进度可改10，想减少输出可改100或0
    parser.add_argument(
        "--log_interval", type=int, default=50,
        help="每隔多少个batch打印一次进度（默认50；设0则只打印每阶段首尾）"
    )

    # --resume: 从checkpoint继续训练
    # 空字符串表示从头训练；auto表示加载checkpoint_dir/last_checkpoint.pth
    # 也可以传入具体文件路径，例如 --resume ./checkpoints/checkpoint_epoch20.pth
    parser.add_argument(
        "--resume", type=str, default="",
        help="断点续训checkpoint路径；填auto则加载checkpoint_dir下的last_checkpoint.pth"
    )

    # --patience: Early Stopping耐心值
    # 如果验证mIoU连续patience轮没有提升，就提前停止，减少过拟合和浪费时间
    # 0表示关闭Early Stopping
    parser.add_argument(
        "--patience", type=int, default=12,
        help="验证mIoU连续多少轮不提升就提前停止；0表示关闭（默认12）"
    )

    # --seed: 固定随机种子
    # 保持模型初始化、数据打乱、部分数据增强更可复现，方便对比实验
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，用于提高训练可复现性（默认42）"
    )

    # --class_weights: CrossEntropy类别权重策略
    # none: 不加权，保持原始CE
    # auto: 从训练mask像素分布估计类别权重，缓解背景/少数类不平衡
    parser.add_argument(
        "--class_weights", type=str, default="auto", choices=["none", "auto"],
        help="CrossEntropy类别权重策略：none不加权，auto按训练mask自动估计（默认auto）"
    )

    # --class_weight_samples: 估计类别权重时抽样多少张mask
    # 抽样可避免每次启动都扫描全量mask；设置0表示扫描全部训练mask
    parser.add_argument(
        "--class_weight_samples", type=int, default=500,
        help="自动类别权重最多抽样多少张mask；0表示扫描全部（默认500）"
    )

    # --max_train_batches / --max_val_batches: 调试烟雾测试专用
    # 默认0表示完整训练/验证，不改变正式训练行为。
    # 新增encoder或排查脚本时可设为2~5，快速确认能前向、反向、验证和保存checkpoint。
    parser.add_argument(
        "--max_train_batches", type=int, default=0,
        help="调试用：每个epoch最多训练多少个batch；0表示完整训练集（默认0）"
    )
    parser.add_argument(
        "--max_val_batches", type=int, default=0,
        help="调试用：每个epoch最多验证多少个batch；0表示完整验证集（默认0）"
    )

    return parser.parse_args()


def create_model(num_classes, encoder):
    """
    创建UNet + 轻量级Encoder分割模型

    使用segmentation_models_pytorch(SMP)库，一行代码创建完整模型

    参数:
        num_classes (int):
            输出类别数，包含背景
            6 = 背景(0) + 划痕(1) + 锈蚀(2) + 压伤(3) + 裂纹(4) + 毛刺(5)
            模型输出6通道，每个通道对应一个类别的概率图

        encoder (str):
            backbone编码器名称
            "mobilenet_v2": 当前SMP环境支持的轻量MobileNet，适合先跑通训练测试
            "efficientnet-b0": 精度和速度较均衡，但部署前需要重新评估算子支持
            注意: 当前安装的SMP 0.5.0不支持"mobilenet_v3_small"

    模型结构:
      Encoder(收缩路径):
        轻量级backbone → 逐层提取特征，空间分辨率从224降到7
        每层输出通过Skip Connection传给Decoder

      Decoder(扩展路径):
        从7x7逐层上采样回224x224
        每层与对应的Skip Connection特征拼接(concat)
        拼接后的特征经卷积融合，恢复空间分辨率

      Skip Connection(跳跃连接):
        把Encoder的中间特征直接传给Decoder
        作用: 保留细小缺陷的边界信息，防止深层网络丢失细节
        这是UNet相比LR-ASPP的关键优势

    smp.Unet 关键参数说明:
      encoder_name:
        backbone名称，SMP支持70+种backbone
        常用: mobilenet_v2, tu-mobilenetv3_small_100.lamb_in1k, efficientnet-b0
      encoder_weights:
        预训练权重，"imagenet"表示用ImageNet分类任务预训练的权重
        预训练backbone已经学会提取边缘/纹理等通用特征
        在小数据集上比从头训练收敛快得多
        None表示不加载预训练，从头训练（不推荐）
      in_channels:
        输入图片通道数，RGB图片=3
        如果用灰度图改为1
      classes:
        输出类别数，决定模型最后一层输出几个通道
    """
    # 先校验encoder名称，避免SMP抛出超长KeyError后看不清真正原因。
    validate_encoder_name(encoder)

    model = smp.Unet(
        encoder_name=encoder,           # backbone编码器
        encoder_weights="imagenet",     # ImageNet预训练权重，加速收敛
        in_channels=3,                  # 输入3通道RGB
        classes=num_classes,            # 输出类别数（由--num_classes控制）
    )
    return model


def compute_iou(preds, targets, num_classes):
    """
    计算每个类别的IoU（Intersection over Union，交并比）

    IoU = 预测正确的像素数 / (预测为该类的像素数 + 真实为该类的像素数 - 交集)
    直觉: 预测区域和真实区域的重叠程度，1.0=完美，0.0=完全没对上

    IoU是语义分割的核心评价指标，比准确率(Accuracy)更有意义
    因为背景像素占绝大多数，Accuracy会被背景拉高，看不出缺陷检测效果

    参数:
        preds (torch.Tensor):
            模型原始输出，形状(B, C, H, W)
            B=batch_size, C=num_classes, H=W=224
            每个像素有C个通道的logits值（未经过softmax）
            argmax后得到每个像素的预测类别

        targets (torch.Tensor):
            真实掩码，形状(B, H, W)
            每个像素值=类别ID（0-5）

        num_classes (int):
            类别数（6）

    返回:
        ious (list[float]):
            每个类别的IoU值，长度=num_classes
            如果某个类别在targets中不存在，返回float('nan')
    """
    preds = preds.argmax(dim=1)  # (B, C, H, W) → (B, H, W)，取每像素最大概率的类别

    ious = []
    for cls in range(num_classes):
        # 交集: 预测为cls 且 真实也是cls 的像素数
        intersection = ((preds == cls) & (targets == cls)).float().sum()
        # 并集: 预测为cls 或 真实为cls 的像素数
        union = ((preds == cls) | (targets == cls)).float().sum()

        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(float('nan'))  # 该类别在targets中不存在，无法计算IoU
    return ious


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes,
                    epoch=None, total_epochs=None, log_interval=50, max_batches=0):
    """
    训练一个epoch（遍历整个训练集一次）

    参数:
        model (nn.Module):
            UNet分割模型

        loader (DataLoader):
            训练集数据加载器，每次迭代返回(batch_images, batch_masks)

        criterion (callable):
            损失函数，输入(pred, target)返回loss值
            这里是CE + 0.5*Dice联合损失

        optimizer (torch.optim.Optimizer):
            AdamW优化器，负责根据梯度更新模型参数

        device (torch.device):
            计算设备，cuda(GPU)或cpu

        num_classes (int):
            类别数（6）

        epoch (int or None):
            当前epoch序号，从1开始；None表示不打印进度，便于单元测试或脚本复用。

        total_epochs (int or None):
            总epoch数；与epoch同时提供时用于终端进度显示。

        log_interval (int):
            batch级进度打印间隔；默认每50个batch打印一次，0表示只打印阶段首尾。

        max_batches (int):
            调试/烟雾测试时最多训练多少个batch；0表示完整遍历训练集。
            这个参数只用于快速验证新encoder能否前向、反向和保存checkpoint，正式训练不要设置。

    返回:
        avg_loss (float): 本epoch平均损失
        miou (float):     本epoch平均mIoU（所有类别IoU的均值）
        avg_ious (dict):  每个类别的平均IoU {class_id: avg_iou}
    """
    model.train()  # 训练模式：启用Dropout和BatchNorm的训练行为

    total_loss = 0
    all_ious = {i: [] for i in range(num_classes)}
    total_batches = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    stage_start_time = time.time()

    for batch_idx, (images, masks) in enumerate(loader, start=1):
        # 将数据搬到GPU（如果有的话）
        images = images.to(device)     # (B, 3, 224, 224)
        masks = masks.to(device)       # (B, 224, 224)

        # 前向推理：图片送入模型，得到预测结果
        outputs = model(images)        # (B, 6, 224, 224)

        # 计算损失：对比预测和真实标签
        loss = criterion(outputs, masks)

        # 反向传播三步曲:
        # 1. 清零上一步的梯度残留
        optimizer.zero_grad()
        # 2. 反向传播，计算当前loss对每个参数的梯度
        loss.backward()
        # 3. 根据梯度更新参数
        optimizer.step()

        total_loss += loss.item()

        # 计算IoU（用于监控训练进度，不参与梯度计算）
        # .detach() 将tensor从计算图中分离，节省显存
        ious = compute_iou(outputs.detach(), masks, num_classes)
        for cls in range(num_classes):
            if ious[cls] == ious[cls]:  # NaN != NaN，所以"不是NaN就记录"
                all_ious[cls].append(ious[cls])

        # 打印batch级进度，避免长epoch期间终端长时间没有任何输出。
        if epoch is not None and total_epochs is not None:
            log_stage_progress(
                stage="训练",
                epoch=epoch,
                total_epochs=total_epochs,
                batch_idx=batch_idx,
                total_batches=total_batches,
                total_loss=total_loss,
                all_ious=all_ious,
                num_classes=num_classes,
                stage_start_time=stage_start_time,
                log_interval=log_interval,
            )

        # 调试模式：只跑指定数量batch，避免为了验证encoder而等待完整epoch。
        # break放在日志之后，保证最后一个被执行的batch也会打印进度。
        if max_batches > 0 and batch_idx >= max_batches:
            break

    # 计算平均损失
    avg_loss = total_loss / total_batches

    # 计算mIoU（所有真实出现过的类别IoU均值）
    # 训练阶段和验证阶段共用summarize_ious，避免统计逻辑不一致。
    miou, avg_ious = summarize_ious(all_ious, num_classes)

    return avg_loss, miou, avg_ious


def validate(model, loader, criterion, device, num_classes,
             epoch=None, total_epochs=None, log_interval=50, max_batches=0):
    """
    验证一个epoch（在验证集上评估模型，不更新参数）

    参数:
        同 train_one_epoch

    返回:
        同 train_one_epoch

    与训练的区别:
      - model.eval(): 关闭Dropout和BatchNorm的训练行为
      - torch.no_grad(): 不计算梯度，节省显存和计算
      - 不调用optimizer，不更新参数

    进度显示:
      epoch和total_epochs同时传入时，会按log_interval打印验证进度。

    max_batches:
      调试/烟雾测试时最多验证多少个batch；0表示完整遍历验证集。
    """
    model.eval()  # 评估模式：关闭Dropout，BatchNorm用全局统计量

    total_loss = 0
    all_ious = {i: [] for i in range(num_classes)}
    total_batches = len(loader) if max_batches <= 0 else min(len(loader), max_batches)
    stage_start_time = time.time()

    with torch.no_grad():  # 不计算梯度，验证不需要反向传播
        for batch_idx, (images, masks) in enumerate(loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            ious = compute_iou(outputs, masks, num_classes)
            for cls in range(num_classes):
                if ious[cls] == ious[cls]:  # 不是NaN就记录
                    all_ious[cls].append(ious[cls])

            # 验证集也打印进度；验证集较大时可以看到评估跑到哪里。
            if epoch is not None and total_epochs is not None:
                log_stage_progress(
                    stage="验证",
                    epoch=epoch,
                    total_epochs=total_epochs,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    total_loss=total_loss,
                    all_ious=all_ious,
                    num_classes=num_classes,
                    stage_start_time=stage_start_time,
                    log_interval=log_interval,
                )

            # 调试模式：只跑指定数量batch，快速验证验证流程和指标统计是否正常。
            if max_batches > 0 and batch_idx >= max_batches:
                break

    avg_loss = total_loss / total_batches

    # 计算验证mIoU时同样保留0分IoU，避免模型漏检某类时指标被虚高。
    miou, avg_ious = summarize_ious(all_ious, num_classes)

    return avg_loss, miou, avg_ious


def main():
    """主训练流程"""
    args = parse_args()
    class_names = build_class_names(args.num_classes)
    set_random_seed(args.seed)
    print(f"随机种子: {args.seed}")

    # ---- 1. 设备选择 ----
    # 优先使用GPU（CUDA），没有GPU时回退到CPU
    # CPU训练极慢，不建议（UNet+MobileNetV3 100 epoch约需2-3天）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # 显示GPU显存信息，方便判断batch_size是否合理
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ---- 2. 创建模型 ----
    model = create_model(args.num_classes, args.encoder)
    model = model.to(device)

    # 打印模型参数量，用于评估部署可行性
    # 轻量backbone的UNet通常为数百万参数，具体数值以这里打印结果为准
    # INT8量化后约2-5MB，STM32MP157可运行
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {param_count:.2f}M")

    # ---- 3. 数据集和数据加载器 ----
    # 训练集：带数据增强 + 预处理
    train_dataset = DefectDataset(
        images_dir=os.path.join(args.data_dir, "train", "images"),
        masks_dir=os.path.join(args.data_dir, "train", "masks"),
        augmentation=get_training_augmentation(),   # 训练集做数据增强
        preprocessing=get_preprocessing(),          # ImageNet归一化
    )

    # 验证集：只做Resize + 预处理，不做随机增强
    val_dataset = DefectDataset(
        images_dir=os.path.join(args.data_dir, "val", "images"),
        masks_dir=os.path.join(args.data_dir, "val", "masks"),
        augmentation=get_validation_augmentation(),  # 只做Resize
        preprocessing=get_preprocessing(),            # ImageNet归一化
    )

    # DataLoader: 从Dataset批量获取数据
    # batch_size:  每次取多少张图片，越大显存占用越多
    # shuffle:     训练集打乱顺序（防止模型学习顺序规律），验证集不打乱
    # num_workers:  数据加载线程数，Windows下设4，报错改0
    # pin_memory:  锁页内存，GPU训练时加速数据传输
    # DataLoader generator控制shuffle随机性；配合seed_worker让多进程增强更可复现。
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,             # 训练集必须打乱
        num_workers=args.num_workers,
        pin_memory=True,          # GPU训练时加速数据传输
        worker_init_fn=seed_worker,
        generator=data_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,            # 验证集不打乱，保证结果可复现
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")

    # ---- 4. 损失函数 ----
    # 使用CE + Dice联合损失，两者互补：
    #
    # CrossEntropyLoss (CE):
    #   像素级分类损失，对每个像素独立计算
    #   优点: 训练稳定，收敛快
    #   缺点: 对小目标缺陷不敏感（背景像素占绝大多数，主导了损失）
    #
    # DiceLoss:
    #   区域重叠度损失，衡量预测区域和真实区域的重叠程度
    #   公式: 1 - 2*|P∩G| / (|P|+|G|)
    #   优点: 对小目标缺陷敏感，不受类别不平衡影响
    #   缺点: 梯度不稳定，单独使用可能训练困难
    #
    # 联合损失 = CE + 0.5*Dice
    # 0.5权重: Dice的梯度波动较大，降低权重保持训练稳定
    # 如果缺陷面积普遍很小，可以增大Dice权重到1.0
    class_weights = None
    if args.class_weights == "auto":
        masks_dir = os.path.join(args.data_dir, "train", "masks")
        class_weights = compute_class_weights_from_masks(
            masks_dir=masks_dir,
            num_classes=args.num_classes,
            max_files=args.class_weight_samples,
        ).to(device)
        weight_text = ", ".join(
            f"{class_names[idx]}={class_weights[idx].item():.3f}"
            for idx in range(args.num_classes)
        )
        print(f"自动类别权重: {weight_text}")
    else:
        print("类别权重: 关闭（CrossEntropyLoss不加权）")

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    def criterion(pred, target):
        """
        联合损失 = CE + 0.5*Dice

        参数:
            pred:   模型输出 (B, C, H, W)
            target: 真实标签 (B, H, W)
        """
        return ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    # ---- 5. 优化器 ----
    # AdamW: Adam的改进版，权重衰减(weight_decay)更合理
    # Adam的权重衰减实际上是L2正则化，AdamW是真正的权重衰减
    # 效果: 防止模型参数过大，减轻过拟合
    # 默认weight_decay=0.01，适合大多数场景
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- 6. 学习率调度器 ----
    # CosineAnnealingLR: 余弦退火调度
    # 学习率从初始值(lr)按余弦曲线慢慢降到最小值(eta_min)
    # 前期lr大，快速学习; 后期lr小，精细调整
    # T_max: 半周期长度，设为总epoch数，整个训练过程是一个余弦周期
    # eta_min: 最小学习率，1e-6接近0，保证最后几乎不更新
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---- 7. TensorBoard日志 ----
    # SummaryWriter 将训练指标写入日志文件
    # 可视化查看: tensorboard --logdir ./logs → 浏览器 localhost:6006
    # 记录内容:
    #   - loss曲线（训练/验证对比）
    #   - mIoU曲线
    #   - 每个类别的IoU变化趋势
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---- 8. 训练循环 ----
    # 用-1初始化，保证第一轮验证即使mIoU=0也会保存一个best_model.pth。
    best_miou = -1.0  # 记录最佳验证mIoU
    start_epoch = 0  # 从第几个epoch开始训练，断点续训时会被checkpoint覆盖
    epochs_without_improve = 0  # Early Stopping计数器，记录验证集连续未提升轮数
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    resume_path = resolve_resume_path(args.resume, args.checkpoint_dir)
    if resume_path is not None:
        start_epoch, best_miou, epochs_without_improve = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=args.num_classes,
            encoder=args.encoder,
        )
        print(
            f"已从checkpoint续训: {resume_path} | "
            f"下一轮: {start_epoch + 1}/{args.epochs} | "
            f"历史最佳mIoU: {best_miou:.4f}"
        )

    if start_epoch >= args.epochs:
        print(
            f"当前checkpoint已经训练到第{start_epoch}轮，"
            f"--epochs={args.epochs} 不会再继续训练。请把--epochs调大。"
        )
        writer.close()
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        # 训练一个epoch
        train_loss, train_miou, train_ious = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.num_classes,
            epoch=epoch + 1,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )

        # 在验证集上评估
        val_loss, val_miou, val_ious = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            epoch=epoch + 1,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
            max_batches=args.max_val_batches,
        )

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印训练结果
        print(f"训练 Loss: {train_loss:.4f} | mIoU: {train_miou:.4f}")
        print(f"验证 Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")
        print(f"学习率: {current_lr:.6f}")

        # 打印每个类别的IoU（重点关注缺陷类别，不要只看mIoU）
        # 背景IoU通常很高(0.9+)，会拉高mIoU，不代表缺陷检测效果好
        for cls_id, cls_name in enumerate(class_names):
            t_iou = train_ious.get(cls_id, 0)
            v_iou = val_ious.get(cls_id, 0)
            print(f"  {cls_name}: train_iou={t_iou:.4f} val_iou={v_iou:.4f}")

        # 写入TensorBoard日志
        writer.add_scalars('loss', {
            'train': train_loss, 'val': val_loss
        }, epoch)
        writer.add_scalars('miou', {
            'train': train_miou, 'val': val_miou
        }, epoch)
        for cls_id, cls_name in enumerate(class_names):
            writer.add_scalar(f'iou/{cls_name}', val_ious.get(cls_id, 0), epoch)

        # 保存最佳模型（验证mIoU超过历史最高时保存）
        # best_model.pth 是最终要用于部署的模型
        # 保存内容:
        #   epoch:              当前epoch编号
        #   model_state_dict:   模型权重（最核心）
        #   optimizer_state_dict:优化器状态（用于恢复训练）
        #   best_miou:          最佳mIoU值
        improved = val_miou > best_miou
        if improved:
            best_miou = val_miou
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            epochs_without_improve = 0
            save_checkpoint(
                path=save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_miou=best_miou,
                num_classes=args.num_classes,
                encoder=args.encoder,
                epochs_without_improve=epochs_without_improve,
            )
            print(f"  ★ 保存最佳模型 mIoU={best_miou:.4f}")
        else:
            epochs_without_improve += 1
            if args.patience > 0:
                print(
                    f"  验证mIoU未提升: {epochs_without_improve}/{args.patience} "
                    f"(历史最佳 {best_miou:.4f})"
                )

        # 每个epoch都保存last_checkpoint，保证Ctrl+C或意外中断后能尽量从最近一轮继续。
        last_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pth")
        save_checkpoint(
            path=last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_miou=best_miou,
            num_classes=args.num_classes,
            encoder=args.encoder,
            epochs_without_improve=epochs_without_improve,
        )

        # 每20轮保存一次检查点
        # 用途: 训练中断时可以从最近的检查点恢复
        # 不需要每个epoch都保存，太占磁盘空间
        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth"
            )
            save_checkpoint(
                path=save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_miou=best_miou,
                num_classes=args.num_classes,
                encoder=args.encoder,
                epochs_without_improve=epochs_without_improve,
            )

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(
                f"\nEarly Stopping: 验证mIoU连续{args.patience}轮没有提升，提前停止训练。"
            )
            break

    writer.close()
    print(f"\n训练完成! 最佳 mIoU = {best_miou:.4f}")
    print(f"最佳模型保存在: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(
        "下一步: "
        f"python export_onnx.py --checkpoint {os.path.join(args.checkpoint_dir, 'best_model.pth')} "
        f"--num_classes {args.num_classes} --encoder {args.encoder}"
    )


if __name__ == "__main__":
    main()
