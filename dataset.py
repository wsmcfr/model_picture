"""
数据集模块：读取图片和掩码，做数据增强，转为PyTorch Tensor

本模块是训练流程的核心数据入口，负责：
1. DefectDataset 类 - 从磁盘读取图片和掩码对，按索引返回Tensor
2. 数据增强函数  - 训练时随机翻转/旋转/加噪声等，增加样本多样性
3. 预处理函数    - ImageNet均值标准差归一化，与预训练backbone匹配

掩码格式说明:
  掩码是单通道PNG图片，每个像素的值 = 类别ID：
    0 = 背景（正常区域，无缺陷）
    1 = 划痕（scratch）
    2 = 锈蚀（rust）
    3 = 压伤（dent）
    4 = 裂纹（crack）
    5 = 毛刺（burr）

  例如：一个像素值为2的掩码像素，表示这个位置是锈蚀缺陷

  为什么用PNG不用JPG?
    - JPG是有损压缩，像素值会被改变（如2变成3），类别就错了
    - PNG是无损的，像素值精确保留，适合做掩码

数据流向:
  磁盘上的jpg+png → DefectDataset.__getitem__() → 数据增强 → 预处理 → Tensor
  → DataLoader批量加载 → 送入模型训练
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# 关闭Albumentations导入时的联网版本检查，避免训练启动阶段因网络超时产生无关警告。
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectDataset(Dataset):
    """
    金属零件缺陷分割数据集

    继承PyTorch的Dataset类，必须实现__getitem__和__len__方法。
    DataLoader会自动调用这两个方法来批量获取训练数据。

    数据目录结构要求:
      images_dir/
      ├── 0000.jpg
      ├── 0001.jpg
      └── ...

      masks_dir/
      ├── 0000.png    ← 与图片同名，后缀不同
      ├── 0001.png    ← 像素值=类别ID（0-5）
      └── ...

    注意事项:
      - 图片和掩码必须同名（后缀可以不同），按名称排序后一一对应
      - 掩码必须是PNG格式，像素值为0-5的整数
      - 良品图片也要有对应的掩码（全0=全背景）
    """

    # 类别定义：像素值 → 类别名
    # 这个列表的索引就是掩码中的像素值
    # 例如 CLASSES[1] = "scratch" 表示像素值1代表划痕
    # 必须与 convert_labelme.py 中的 LABEL_MAP 一致
    CLASSES = [
        "background",   # 0 - 背景（正常区域，无缺陷）
        "scratch",      # 1 - 划痕（表面线性划伤）
        "rust",         # 2 - 锈蚀（表面氧化锈斑）
        "dent",         # 3 - 压伤（凹陷、磕碰痕迹）
        "crack",        # 4 - 裂纹（细小裂纹）
        "burr",         # 5 - 毛刺（边缘冲压毛刺）
    ]

    def __init__(self, images_dir, masks_dir, classes=None,
                 augmentation=None, preprocessing=None):
        """
        初始化数据集

        参数:
            images_dir (str):
                图片目录路径，目录下放 .jpg/.png 格式的原图
                例如: "./datasets/train/images"

            masks_dir (str):
                掩码目录路径，目录下放 .png 格式的掩码图
                掩码像素值=类别ID（0=背景, 1=划痕, ...）
                例如: "./datasets/train/masks"

            classes (list[str] or None):
                要使用的类别列表，None表示使用全部6类
                如果只想训练部分类别，可以指定子集，例如:
                  classes=["scratch", "rust"]  → 只检测划痕和锈蚀
                其他类别会被映射到背景(0)
                默认None = 使用全部6类（背景+5种缺陷）

            augmentation (albumentations.Compose or None):
                数据增强变换，训练时传入get_training_augmentation()
                验证/测试时传入None或get_validation_augmentation()
                albumentations库会同时对图片和掩码做相同的几何变换
                （翻转、旋转等），保证图片和掩码的对应关系不被破坏

            preprocessing (albumentations.Compose or None):
                预处理变换，通常是ImageNet均值标准差归一化
                传入get_preprocessing()
                必须与模型预训练时的归一化方式一致，否则推理结果会偏
        """
        # 获取所有图片文件路径（按名称排序保证一致性）
        # 排序很重要！因为图片和掩码是按文件名一一对应的
        # 支持 .jpg/.jpeg/.png/.bmp 四种图片格式
        self.images_fps = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        # 获取所有掩码文件路径（按名称排序）
        # 掩码只支持 .png/.bmp，因为需要无损格式保留精确像素值
        self.masks_fps = sorted([
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.endswith(('.png', '.bmp'))
        ])

        # 检查图片和掩码数量是否一致
        # 如果不一致，说明数据有问题（漏标注、文件丢失等）
        assert len(self.images_fps) == len(self.masks_fps), \
            f"图片数({len(self.images_fps)})和掩码数({len(self.masks_fps)})不一致，请检查数据"

        # 类别映射（如果只训练部分类别）
        # class_map: {原始类别ID → 新类别ID}
        # 例如只训练scratch和rust:
        #   {0: 0, 1: 1, 2: 2} → 背景还是0, scratch变1, rust变2
        #   其他类别（dent/crack/burr）的像素会被映射到0（背景）
        if classes is not None:
            self.class_map = {0: 0}  # 背景永远是0
            for cls_name in classes:
                cls_id = self.CLASSES.index(cls_name)     # 原始类别ID
                self.class_map[cls_id] = len(self.class_map)  # 新类别ID（递增）
        else:
            self.class_map = None  # None表示使用全部类别，不做映射

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """
        按索引读取一对(图片, 掩码)，返回(image_tensor, mask_tensor)

        参数:
            i (int): 数据索引，0到len(dataset)-1

        返回:
            image (torch.Tensor): 图片张量，形状(3, H, W)，float32，ImageNet归一化后
            mask  (torch.Tensor): 掩码张量，形状(H, W)，long类型，像素值=类别ID

        数据处理流程:
          1. 读取原图 → RGB三通道numpy数组 (H, W, 3)
          2. 读取掩码 → 单通道numpy数组 (H, W)
          3. 类别映射（如果只训练部分类别）
          4. 数据增强（翻转/旋转/噪声等，图片和掩码同步变换）
          5. 预处理（ImageNet均值标准差归一化，只对图片）
          6. 转为PyTorch Tensor
        """
        # 读取图片，强制转为RGB三通道
        # convert("RGB") 确保即使是灰度图也会变成3通道
        # np.array() 将PIL Image转为numpy数组，形状(H, W, 3)，uint8
        image = np.array(Image.open(self.images_fps[i]).convert("RGB"))

        # 读取掩码，单通道
        # 掩码的像素值就是类别ID（0-5），必须是整数
        # np.array() 形状(H, W)，uint8
        mask = np.array(Image.open(self.masks_fps[i]))

        # 如果掩码是RGB三通道（某些标注工具会生成3通道掩码）
        # 取第一个通道即可，三个通道值相同
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # 类别映射（如果只训练部分类别）
        # 将不在训练列表中的类别映射到背景(0)
        if self.class_map is not None:
            new_mask = np.zeros_like(mask)    # 全0初始化（背景）
            for old_id, new_id in self.class_map.items():
                new_mask[mask == old_id] = new_id
            mask = new_mask

        # 数据增强
        # albumentations同时对image和mask做相同的几何变换
        # 例如水平翻转：图片和掩码同时翻转，保证对应关系不被破坏
        # p=0.5 表示每次有50%概率执行该变换
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 预处理（ImageNet归一化）
        # 只对图片做归一化，掩码不做（掩码像素值是类别ID，不能改）
        # 归一化公式: pixel = (pixel/255 - mean) / std
        # mean和std是ImageNet数据集统计值，与预训练backbone匹配
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 转为PyTorch Tensor
        #
        # image: (H, W, 3) → (3, H, W)
        #   permute(2, 0, 1) 把通道维从最后移到最前
        #   .float() 转为float32，PyTorch模型需要float输入
        #
        # mask: (H, W) 保持不变
        #   .long() 转为int64，CrossEntropyLoss要求标签是long类型
        #   不做one-hot编码，PyTorch的CE损失函数内部会处理
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        """返回数据集大小（图片数量）"""
        return len(self.images_fps)


# ============================================================
# 数据增强定义
# ============================================================

def get_training_augmentation():
    """
    训练集数据增强

    目的：通过随机变换增加训练样本的多样性，防止模型过拟合。
    过拟合的表现：训练集准确率很高，但验证集/实际效果很差。
    数据增强让模型见到更多"变体"，学到更鲁棒的特征。

    为什么需要这些增强？
      - 翻转/旋转：实际工位上零件放置角度会有微小偏差
      - 模糊/噪声：摄像头在不同条件下可能有轻微离焦或传感器噪声
      - 亮度变化：工位光照可能随时间略有波动
      - 随机遮挡：防止模型依赖局部特征（只看一个角就能判断）

    注意：数据增强只对训练集使用，验证集和测试集不做增强！

    每个增强操作的p参数表示执行概率：
      p=0.5 → 每次有50%概率执行该变换
      p=0.3 → 每次有30%概率执行
      p=1.0 → 每次都执行（如Resize）

    albumentations库的关键特性：
      对image和mask做同步的几何变换（翻转、旋转、缩放等）
      保证图片和掩码的空间对应关系不被破坏
    """
    train_transform = [
        # ---- 几何变换 ----

        # 水平翻转（左右镜像）
        # 模拟零件左右放置差异，划痕方向也会随之翻转
        # p=0.5: 每次有50%概率翻转
        A.HorizontalFlip(p=0.5),

        # 垂直翻转（上下镜像）
        # 模拟零件上下放置差异
        A.VerticalFlip(p=0.5),

        # 随机旋转90度的整数倍（0°/90°/180°/270°）
        # 模拟零件旋转放置
        # 用90度倍数是因为掩码是像素级的，非90度旋转会导致插值误差
        A.RandomRotate90(p=0.5),

        # 小幅度平移+缩放+旋转的组合变换
        # 一次操作同时做三种变换，模拟零件位置微偏
        # translate_percent=(-0.05, 0.05): 平移幅度不超过图片宽高的5%（约11像素@224）
        # scale=(0.9, 1.1):                缩放幅度在0.9x~1.1x之间
        # rotate=(-15, 15):                旋转角度在-15°~+15°之间
        # border_mode=0:    超出边界的区域用0（黑色/背景）填充
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=0,
            fill=0,
            fill_mask=0,
            p=0.5
        ),

        # ---- 图像质量变换（模拟拍摄条件变化）----

        # 高斯模糊
        # 模拟摄像头轻微离焦/抖动导致的模糊
        # blur_limit=3: 模糊核大小最大3x3
        # p=0.3: 30%概率执行，不要太频繁，否则模型学不到清晰边界
        A.GaussianBlur(blur_limit=3, p=0.3),

        # 高斯噪声
        # 模拟摄像头传感器的电子噪声
        # std_range=(0.02, 0.08): 噪声标准差占像素值范围的2%-8%，值越大噪声越强
        # mean_range=(0.0, 0.0): 噪声均值保持为0，只增加随机波动，不整体改变亮度
        # p=0.3: 30%概率执行
        A.GaussNoise(
            std_range=(0.02, 0.08),
            mean_range=(0.0, 0.0),
            p=0.3
        ),

        # 随机亮度对比度
        # 模拟工位光照的日间波动
        # brightness_limit=0.2: 亮度在0.8x~1.2x之间变化
        # contrast_limit=0.2:  对比度在0.8x~1.2x之间变化
        # p=0.5: 50%概率执行，光照变化是很常见的
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # CLAHE（对比度受限自适应直方图均衡）
        # 增强局部对比度，让细微缺陷（裂纹、小压伤）更清晰
        # 与普通直方图均衡不同，CLAHE按局部区域做，不会过度放大噪声
        # p=0.3: 30%概率执行
        A.CLAHE(p=0.3),

        # ---- 正则化变换（防止过拟合）----

        # 随机遮挡（CoarseDropout / Cutout）
        # 在图片上随机生成矩形黑块，遮挡部分区域
        # 强制模型不依赖局部特征，学会从整体判断
        # num_holes_range=(1, 8): 最少遮挡1块，最多遮挡8块
        # hole_height_range=(8, 16), hole_width_range=(8, 16): 每个遮挡区域8-16像素
        # fill=0: 图片遮挡区域填充0（黑色）
        # fill_mask=0: 掩码遮挡区域填充0，表示遮挡处不再强迫模型学习原缺陷标签
        # p=0.3: 30%概率执行
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0,
            fill_mask=0,
            p=0.3
        ),

        # ---- 尺寸归一化（必须放最后）----

        # 缩放到固定224x224
        # 224x224是MobileNetV3-Small的标准输入尺寸
        # 必须与部署时NCNN推理的输入尺寸一致
        # 如果改了这个尺寸，部署时也要对应修改
        # p=1.0（隐含，Resize总是执行）
        A.Resize(224, 224),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """
    验证集/测试集数据增强

    验证集不做任何随机增强，只做Resize到224x224。
    原因：验证集要反映模型在真实数据上的表现，不能"作弊"。
    如果验证集也做随机增强，每次验证结果都不同，无法比较。

    只做Resize是因为：
      1. 模型输入要求固定224x224，必须缩放
      2. 其他变换都是随机的，验证时不应该引入随机性
    """
    return A.Compose([
        A.Resize(224, 224),
    ])


def get_preprocessing():
    """
    预处理：ImageNet均值标准差归一化

    归一化公式（对每个通道独立计算）：
      normalized = (pixel / 255 - mean) / std

    为什么用ImageNet的均值和标准差？
      MobileNetV3-Small的backbone是在ImageNet上预训练的
      预训练时用的就是这个均值和标准差
      如果训练时不做同样的归一化，预训练权重就不匹配
      模型需要重新学习输入分布，收敛会慢很多

    mean = [0.485, 0.456, 0.406]  (R, G, B 三通道均值)
    std  = [0.229, 0.224, 0.225]  (R, G, B 三通道标准差)

    注意：这个归一化只对image做，mask不做！
    mask的像素值是类别ID（0-5），归一化就失去意义了

    部署时也要做同样的归一化！
    NCNN推理代码中的 substract_mean_normalize 参数必须与这里一致
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],    # ImageNet RGB均值（R, G, B）
            std=[0.229, 0.224, 0.225],      # ImageNet RGB标准差（R, G, B）
        ),
    ])
