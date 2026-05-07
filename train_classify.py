"""
MobileNetV3-Small 金属零件良坏分类训练脚本

项目背景:
  为 STM32MP157 工业缺陷检测系统训练第一阶段主链路分类模型
  输入: (1, 3, 224, 224) RGB 零件图片
  输出: 二分类概率 [good, bad]

模型:
  骨干: torchvision.models.mobilenet_v3_small (ImageNet 预训练)
  分类头: 修改最后全连接层为 2 类输出
  参数量: ~2.5M, 非常适合 MP157 CPU 推理

数据组织:
  datasets_classify/
  ├── train/
  │   ├── good/        ← 良品图片
  │   └── bad/         ← 缺陷图片
  ├── val/
  │   ├── good/
  │   └── bad/

用法:
  标准训练:
    D:\model_picture\defect-unet\python.exe train_classify.py --data_dir ./datasets_classify --epochs 100 --batch_size 16

  显存不够:
    D:\model_picture\defect-unet\python.exe train_classify.py --batch_size 8

  断点续训:
    D:\model_picture\defect-unet\python.exe train_classify.py --resume auto

  监控:
    D:\model_picture\defect-unet\python.exe -m tensorboard.main --logdir ./logs_classify
"""

import os
import argparse
import time
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms, datasets

import numpy as np


# ============================================================
# 随机种子与可复现性
# ============================================================

def set_random_seed(seed):
    """
    固定训练过程中的主要随机源，减少每次运行之间的随机波动

    参数:
        seed (int): 随机种子；相同代码、数据和环境下，种子一致可提高结果可复现性

    返回:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 这两个设置会略微牺牲一点速度，但更适合调试和对比实验
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """
    为 DataLoader 的每个 worker 设置独立但可复现的随机种子

    参数:
        worker_id (int): DataLoader 传入的 worker 编号

    返回:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 模型构建
# ============================================================

def build_model(num_classes=2, pretrained=True):
    """
    构建 MobileNetV3-Small 分类模型

    参数:
        num_classes (int): 输出类别数，默认 2 (good/bad)
        pretrained (bool): 是否加载 ImageNet 预训练权重

    返回:
        model (nn.Module): 构建好的模型

    说明:
        torchvision 的 mobilenet_v3_small 分类头结构:
        Sequential(
            Linear(576, 1024), Hardswish, Dropout(p=0.2), Linear(1024, num_classes)
        )
        我们只需要修改最后一层 Linear 的输出维度
    """
    # 加载 torchvision 官方的 mobilenet_v3_small
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.mobilenet_v3_small(weights=weights)

    # 修改分类头: 原 1024 -> num_classes 类
    # classifier[-1] 是最后一个 Linear 层
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# 数据预处理
# ============================================================

def get_data_transforms():
    """
    获取训练和验证的数据预处理方法

    返回:
        train_transform (Compose): 训练用变换（含数据增强）
        val_transform (Compose): 验证用变换（不含增强）

    说明:
        使用 ImageNet 预训练模型的标准归一化参数:
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 训练变换: 含随机翻转、旋转、颜色抖动等增强
    # 注意: 分类任务直接缩放到 224x224，不保持比例
    # 原因:
    #   1. 近距离拍摄时零件可能占满画面，保持比例缩放会导致零件在 224x224 中过小
    #   2. 分类任务对轻微变形不敏感， Resize 能保证零件在输入中足够大
    #   3. 远距离有边距的图，Resize 后零件仍然清晰可见
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),          # 直接缩放到 224x224，零件占满输入
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            transforms.RandomRotation(degrees=15),   # 随机旋转 ±15 度
            transforms.ColorJitter(                 # 颜色抖动，增强泛化能力
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),                     # PIL -> Tensor [0,1]
            normalize,                                 # ImageNet 归一化
        ]
    )

    # 验证变换: 与训练保持一致，直接 Resize(224,224)
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


# ============================================================
# 训练与验证函数
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None,
                    total_epochs=None, log_interval=50):
    """
    训练一个 epoch

    参数:
        model (nn.Module): 待训练模型
        loader (DataLoader): 训练数据加载器
        criterion (nn.Module): 损失函数
        optimizer (optim.Optimizer): 优化器
        device (torch.device): 计算设备 (cuda/cpu)
        epoch (int): 当前 epoch 编号
        total_epochs (int): 总 epoch 数
        log_interval (int): 每隔多少个 batch 打印一次进度

    返回:
        avg_loss (float): 该 epoch 的平均损失
        avg_acc (float): 该 epoch 的平均准确率 (%)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_batches = len(loader)

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        # 数据搬运到 GPU/CPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # 进度打印
        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = correct / total_samples * 100.0
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
            print(
                f"  [Train] Batch {batch_idx + 1}/{num_batches} | "
                f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | "
                f"ETA: {eta:.1f}s"
            )

    avg_loss = total_loss / num_batches
    avg_acc = correct / total_samples * 100.0
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    """
    在验证集上评估模型

    参数:
        model (nn.Module): 待评估模型
        loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数
        device (torch.device): 计算设备

    返回:
        avg_loss (float): 验证平均损失
        avg_acc (float): 验证平均准确率 (%)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total_samples * 100.0
    return avg_loss, avg_acc


# ============================================================
# 辅助函数
# ============================================================

def resolve_resume_path(resume, checkpoint_dir):
    """
    解析续训路径

    参数:
        resume (str): 命令行 --resume 值
        checkpoint_dir (str): checkpoint 保存目录

    返回:
        str or None: checkpoint 路径；None 表示从头训练
    """
    if not resume:
        return None
    if resume.lower() == "auto":
        return os.path.join(checkpoint_dir, "last_checkpoint.pth")
    return resume


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, num_classes):
    """
    保存训练状态

    参数:
        path (str): 保存路径
        epoch (int): 当前 epoch
        model (nn.Module): 模型
        optimizer (optim.Optimizer): 优化器
        scheduler (lr_scheduler): 学习率调度器
        best_acc (float): 最佳验证准确率
        num_classes (int): 类别数

    返回:
        None
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
        "num_classes": num_classes,
    }
    torch.save(checkpoint, path)


# ============================================================
# 主训练入口
# ============================================================

def main():
    """
    主训练入口
    """
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small 金属零件良坏分类训练"
    )

    parser.add_argument(
        "--data_dir", type=str, default="./datasets_classify",
        help="分类数据集根目录（需含 train/good, train/bad, val/good, val/bad）"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="类别数，默认 2 (good/bad)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="批大小（RTX 4060 8GB 可用 16，显存不够改 8）"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练总轮数"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="初始学习率"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints_classify",
        help="模型保存目录"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs_classify",
        help="TensorBoard 日志目录"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="数据加载线程数（报 BrokenPipeError 改 0）"
    )
    parser.add_argument(
        "--log_interval", type=int, default=50,
        help="每隔多少个 batch 打印进度"
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="断点续训路径；填 auto 则自动加载 last_checkpoint.pth"
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="验证准确率连续多少轮不提升就提前停止；0 表示关闭"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--pretrained", type=int, default=1, choices=[0, 1],
        help="是否加载 ImageNet 预训练权重（1=是，0=否）"
    )

    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据变换
    train_transform, val_transform = get_data_transforms()

    # 加载数据集
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"数据集目录不存在: {train_dir} 或 {val_dir}\n"
            f"请按以下结构组织分类数据集:\n"
            f"  {args.data_dir}/\n"
            f"    train/good/   ← 良品训练图\n"
            f"    train/bad/    ← 缺陷训练图\n"
            f"    val/good/     ← 良品验证图\n"
            f"    val/bad/      ← 缺陷验证图"
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"类别映射: {train_dataset.class_to_idx}")

    # DataLoader
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 构建模型
    model = build_model(num_classes=args.num_classes, pretrained=bool(args.pretrained))
    model = model.to(device)

    # 损失函数: 类别不平衡时自动加权
    # 计算每个类别的样本数，给少数类更高权重
    class_counts = [0] * args.num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = torch.tensor(
        [total / (args.num_classes * c) if c > 0 else 1.0 for c in class_counts],
        dtype=torch.float32
    ).to(device)
    print(f"类别样本数: {class_counts}, 自动权重: {class_weights.cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度: 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # 续训处理
    start_epoch = 1
    best_acc = 0.0
    resume_path = resolve_resume_path(args.resume, args.checkpoint_dir)

    if resume_path and os.path.exists(resume_path):
        print(f"从检查点恢复: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"恢复完成，从 epoch {start_epoch} 继续，当前最佳验证准确率: {best_acc:.2f}%")
    else:
        if resume_path:
            print(f"警告: 检查点不存在: {resume_path}，将从头训练")

    # Early Stopping
    epochs_without_improve = 0
    patience = args.patience

    # 训练循环
    print(f"\n===== 开始训练 =====")
    print(f"总轮数: {args.epochs}, 批次大小: {args.batch_size}, 学习率: {args.lr}")
    print(f"类别数: {args.num_classes}, 随机种子: {args.seed}")
    print(f"=" * 50)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, total_epochs=args.epochs, log_interval=args.log_interval
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 日志
        print(
            f"\n===== Epoch {epoch}/{args.epochs} | 耗时: {epoch_time:.1f}s =====\n"
            f"  训练 Loss: {train_loss:.4f} | 训练 Acc: {train_acc:.2f}%\n"
            f"  验证 Loss: {val_loss:.4f} | 验证 Acc: {val_acc:.2f}%\n"
            f"  学习率: {current_lr:.6f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_model.pth"),
                epoch, model, optimizer, scheduler, best_acc, args.num_classes
            )
            print(f"  ★ 保存最佳模型 (val_acc={val_acc:.2f}%)")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            print(f"  未改善: {epochs_without_improve}/{patience} 轮")

        # 每 20 轮保存检查点
        if epoch % 20 == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pth"),
                epoch, model, optimizer, scheduler, best_acc, args.num_classes
            )

        # 保存 last_checkpoint（用于续训）
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "last_checkpoint.pth"),
            epoch, model, optimizer, scheduler, best_acc, args.num_classes
        )

        # Early Stopping
        if patience > 0 and epochs_without_improve >= patience:
            print(f"\n!!! Early Stopping: 连续 {patience} 轮验证准确率未提升")
            print(f"    最佳验证准确率: {best_acc:.2f}%")
            break

    # 训练结束
    writer.close()
    print(f"\n===== 训练完成 =====")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"最佳模型: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
