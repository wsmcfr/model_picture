"""
Labelme标注转换脚本

功能:
  1. convert命令: 将Labelme标注的JSON文件转换为训练用的掩码PNG图片
  2. split命令:  将转换后的数据集按比例划分为train/val/test

为什么需要转换?
  Labelme保存的是JSON格式的多边形坐标，模型无法直接使用
  模型需要的是掩码图片: 每个像素的值=类别ID (0-5)
  本脚本将JSON中的多边形"填充"成掩码图片

掩码格式说明:
  掩码是单通道PNG图片，像素值=类别ID
    0 = 背景
    1 = 划痕
    2 = 锈蚀
    3 = 压伤
    4 = 裂纹
    5 = 毛刺
  必须用PNG不用JPG，因为JPG有损压缩会改变像素值

用法:
  步骤1 - 转换标注（Labelme JSON → 掩码PNG）:
    python convert_labelme.py convert <json目录> <输出图目录> <输出掩码目录>
    示例:
      python convert_labelme.py convert ./raw_images ./converted_images ./converted_masks

  步骤2 - 划分数据集（按7:1.5:1.5分为train/val/test）:
    python convert_labelme.py split <图目录> <掩码目录> <输出根目录>
    示例:
      python convert_labelme.py split ./converted_images ./converted_masks ./datasets

完整流程:
  Labelme标注(JSON) → convert → 掩码PNG → split → datasets/train|val|test
"""

import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict


# 类别名称 → 像素值映射
# 必须与dataset.py中的CLASSES列表一致
# Labelme标注时用的标签名必须在这个映射表中，否则会被跳过
# 如果需要增加新的缺陷类型，在这里添加对应的像素值
LABEL_MAP = {
    "background": 0,    # 背景（正常区域）
    "scratch":    1,    # 划痕（表面线性划伤）
    "rust":       2,    # 锈蚀（表面氧化锈斑）
    "dent":       3,    # 压伤（凹陷、磕碰痕迹）
    "crack":      4,    # 裂纹（细小裂纹）
    "burr":       5,    # 毛刺（边缘冲压毛刺）
}


def json_to_mask(json_path):
    """
    将单个Labelme JSON文件转换为掩码数组

    Labelme JSON文件结构:
      {
        "imageHeight": 480,        ← 图像高度
        "imageWidth": 640,         ← 图像宽度
        "imageData": "base64...",  ← 内嵌的base64编码原图（可能为null）
        "imagePath": "0001.jpg",   ← 原图文件名（imageData为null时使用）
        "shapes": [                ← 标注的多边形列表
          {
            "label": "scratch",    ← 标签名（必须与LABEL_MAP的key匹配）
            "points": [[x1,y1], [x2,y2], ...],  ← 多边形顶点坐标
            "shape_type": "polygon"
          },
          ...
        ]
      }

    参数:
        json_path (str): Labelme JSON文件的完整路径

    返回:
        image (numpy.ndarray): 原图数组，形状(H, W, 3)，RGB格式，uint8
        mask (numpy.ndarray):  掩码数组，形状(H, W)，像素值=类别ID，uint8

    处理逻辑:
      1. 读取JSON，获取图像尺寸
      2. 创建全0掩码（0=背景）
      3. 遍历shapes列表，对每个多边形:
         - 检查标签是否在LABEL_MAP中
         - 用cv2.fillPoly将多边形区域填充为对应的类别ID
      4. 如果多个多边形重叠，后绘制的覆盖先绘制的
         （通常不会重叠，除非同一区域标了两种缺陷）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取图像尺寸，用于创建对应大小的掩码
    img_height = data["imageHeight"]
    img_width = data["imageWidth"]

    # 创建空白掩码（全0=背景）
    # dtype=uint8: 像素值范围0-255，足够存0-5的类别ID
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 获取原图
    if "imageData" in data and data["imageData"] is not None:
        # JSON内嵌了base64编码的原图（Labelme默认设置）
        # 解码base64 → 二进制数据 → PIL Image → numpy数组
        import base64
        import io
        img_data = base64.b64decode(data["imageData"])
        image = np.array(Image.open(io.BytesIO(img_data)).convert("RGB"))
    else:
        # imageData为空（Labelme设置中关闭了"Store data to JSON"）
        # 需要从JSON同目录读取原图文件
        # imagePath 是原图的相对路径（相对于JSON文件所在目录）
        img_path = os.path.join(
            os.path.dirname(json_path),
            data["imagePath"]
        )
        image = np.array(Image.open(img_path).convert("RGB"))

    # 遍历所有标注形状，将多边形填充到掩码中
    for shape in data["shapes"]:
        label = shape["label"]       # 标签名，如"scratch"
        points = shape["points"]     # 多边形顶点坐标列表 [[x1,y1],[x2,y2],...]

        # 检查标签是否在映射表中
        # 如果用了不在LABEL_MAP中的标签（如中文标签或拼写错误），跳过并警告
        if label not in LABEL_MAP:
            print(f"警告: 未知标签 '{label}'，跳过（可用标签: {list(LABEL_MAP.keys())})")
            continue

        class_id = LABEL_MAP[label]

        # 将多边形点坐标转为numpy数组
        # points格式: [[x1,y1], [x2,y2], ...] → shape: (N, 2)
        # dtype必须为int32，cv2.fillPoly要求整数坐标
        pts = np.array(points, dtype=np.int32)

        # 用OpenCV的fillPoly填充多边形
        # fillPoly(mask, [pts], color):
        #   mask:  掩码数组，会被直接修改
        #   [pts]: 多边形顶点列表（注意是列表的列表，支持多个多边形）
        #   color: 填充值，这里是类别ID（1-5）
        # 填充后的效果：多边形内部所有像素被设为class_id
        import cv2
        cv2.fillPoly(mask, [pts], class_id)

    return image, mask


def convert_dataset(json_dir, output_img_dir, output_mask_dir):
    """
    批量转换Labelme JSON为掩码PNG

    从json_dir读取所有.json文件，逐个转换为图片和掩码，保存到输出目录

    参数:
        json_dir (str):
            Labelme JSON文件所在目录
            通常与原图在同一目录（Labelme默认把JSON保存在图片旁边）
            例如: ./raw_images/
            目录下有: 0001.jpg, 0001.json, 0002.jpg, 0002.json, ...

        output_img_dir (str):
            输出原图目录
            从JSON中提取的原图会被保存为.jpg格式
            例如: ./converted_images/
            保存为: 0001.jpg, 0002.jpg, ...

        output_mask_dir (str):
            输出掩码目录
            转换后的掩码会被保存为.png格式
            例如: ./converted_masks/
            保存为: 0001.png, 0002.png, ...
            掩码像素值=类别ID（0-5），用PNG无损格式保留精确值

    注意:
      - 保存的原图是JPEG格式(quality=95)，因为原图不需要精确像素值
      - 保存的掩码是PNG格式，因为像素值是类别ID，必须精确保留
      - 如果用JPG保存掩码，压缩会改变像素值(如2→3)，类别就错了
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 找到所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"找到 {len(json_files)} 个JSON标注文件")

    # 如果没有JSON文件，可能是目录路径错误
    if len(json_files) == 0:
        print(f"错误: 在 {json_dir} 下没有找到.json文件")
        print("请确认Labelme标注文件保存在正确的目录")
        return

    success = 0
    fail = 0

    for fname in json_files:
        try:
            json_path = os.path.join(json_dir, fname)
            image, mask = json_to_mask(json_path)

            # 保存原图（JPEG格式，quality=95）
            # quality=95: 推荐的质量，文件大小和画质的平衡
            # basename: 去掉.json后缀，作为输出文件名
            # 例如: 0001.json → 0001.jpg
            basename = os.path.splitext(fname)[0]
            img_out = os.path.join(output_img_dir, basename + ".jpg")
            Image.fromarray(image).save(img_out, quality=95)

            # 保存掩码（PNG格式，保留精确像素值）
            # PNG无损格式，像素值不会被改变
            # 不需要指定quality参数，PNG本身是无损的
            mask_out = os.path.join(output_mask_dir, basename + ".png")
            Image.fromarray(mask).save(mask_out)

            success += 1
            if success % 50 == 0:
                print(f"已转换 {success}/{len(json_files)}")

        except Exception as e:
            print(f"转换失败 {fname}: {e}")
            fail += 1

    print(f"\n转换完成: 成功{success}, 失败{fail}")

    if fail > 0:
        print("失败原因通常是:")
        print("  - JSON文件损坏或格式不正确")
        print("  - JSON引用的原图文件不存在")
        print("  - 标签名不在LABEL_MAP中")


def split_dataset(img_dir, mask_dir, output_base_dir, ratios=(0.7, 0.15, 0.15)):
    """
    将数据集按比例随机划分为train/val/test

    为什么要划分?
      - train(训练集): 用于训练模型，模型看这些数据学习参数
      - val(验证集):   训练过程中评估模型，用于选择最佳模型和调整超参数
      - test(测试集):  训练完成后最终评估，报告模型的实际效果

    三个集必须互不重叠，否则评估结果不可信（数据泄露）

    参数:
        img_dir (str):
            原图目录，包含所有转换后的.jpg图片
            例如: ./converted_images/

        mask_dir (str):
            掩码目录，包含所有转换后的.png掩码
            例如: ./converted_masks/

        output_base_dir (str):
            输出根目录，划分后的数据集保存在这里
            例如: ./datasets/
            自动生成子目录:
              datasets/train/images/, datasets/train/masks/
              datasets/val/images/,   datasets/val/masks/
              datasets/test/images/,  datasets/test/masks/

        ratios (tuple):
            (train, val, test) 比例，默认(0.7, 0.15, 0.15)
            即70%训练 + 15%验证 + 15%测试
            常见比例:
              - (0.7, 0.15, 0.15): 标准划分
              - (0.8, 0.1, 0.1):   数据量少时，多分一些给训练
              - (0.6, 0.2, 0.2):   数据量多时，多分一些给验证和测试

    划分逻辑:
      1. 获取图片和掩码的文件名集合
      2. 取交集（同时有图片和掩码的才是有效样本）
      3. 固定随机种子(seed=42)打乱顺序
         固定种子保证每次运行划分结果一致（可复现）
      4. 按比例切分为三个子集
      5. 复制文件到对应目录

    为什么用seed=42?
      42是机器学习领域常用的随机种子（源自《银河系漫游指南》）
      固定种子的好处:
        - 每次划分结果一致，方便对比不同训练实验
        - 如果训练出问题，可以用相同的数据划分复现
    """
    import shutil
    import random

    # 获取所有图片文件名（去掉后缀，只保留basename）
    # 用set去重，避免重复文件名
    img_names = set()
    for f in os.listdir(img_dir):
        basename = os.path.splitext(f)[0]
        img_names.add(basename)

    # 获取所有掩码文件名（去掉后缀）
    mask_names = set()
    for f in os.listdir(mask_dir):
        basename = os.path.splitext(f)[0]
        mask_names.add(basename)

    # 取交集：同时有图片和掩码的样本才是有效的
    # 如果只有图片没有掩码（忘记标注），或者反过来，都跳过
    valid_names = sorted(list(img_names & mask_names))
    print(f"有效样本数: {len(valid_names)}")

    if len(valid_names) == 0:
        print("错误: 没有找到同时有图片和掩码的样本！")
        print(f"  图片目录: {img_dir} ({len(img_names)} 张)")
        print(f"  掩码目录: {mask_dir} ({len(mask_names)} 张)")
        print("  请确认两个目录的文件名（不含后缀）是一一对应的")
        return

    # 随机打乱（固定种子保证可复现）
    # 每次运行都会得到相同的划分结果
    # 如果想换一种划分，改seed值即可
    random.seed(42)
    random.shuffle(valid_names)

    # 计算划分点
    # 例如1000个样本:
    #   n_train = 1000 * 0.7 = 700
    #   n_val   = 1000 * 0.15 = 150
    #   n_test  = 剩余 150
    n = len(valid_names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_names = valid_names[:n_train]                         # 前70%
    val_names = valid_names[n_train:n_train + n_val]           # 中间15%
    test_names = valid_names[n_train + n_val:]                 # 最后15%

    print(f"划分: train={len(train_names)}, val={len(val_names)}, test={len(test_names)}")

    # 创建目录并复制文件
    splits = {"train": train_names, "val": val_names, "test": test_names}
    for split_name, names in splits.items():
        split_img_dir = os.path.join(output_base_dir, split_name, "images")
        split_mask_dir = os.path.join(output_base_dir, split_name, "masks")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        for name in names:
            # 复制图片（尝试多种后缀，因为原图可能是jpg/png/bmp）
            # 统一输出为.jpg格式
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                src = os.path.join(img_dir, name + ext)
                if os.path.exists(src):
                    dst = os.path.join(split_img_dir, name + ".jpg")
                    shutil.copy2(src, dst)    # copy2保留文件元信息（时间戳等）
                    break

            # 复制掩码（尝试png和bmp后缀）
            # 统一输出为.png格式
            for ext in ['.png', '.bmp']:
                src = os.path.join(mask_dir, name + ext)
                if os.path.exists(src):
                    dst = os.path.join(split_mask_dir, name + ".png")
                    shutil.copy2(src, dst)
                    break

    print(f"数据集已保存到: {output_base_dir}")
    print(f"目录结构:")
    print(f"  {output_base_dir}/train/images/ ({len(train_names)} 张)")
    print(f"  {output_base_dir}/train/masks/  ({len(train_names)} 张)")
    print(f"  {output_base_dir}/val/images/   ({len(val_names)} 张)")
    print(f"  {output_base_dir}/val/masks/    ({len(val_names)} 张)")
    print(f"  {output_base_dir}/test/images/  ({len(test_names)} 张)")
    print(f"  {output_base_dir}/test/masks/   ({len(test_names)} 张)")
    print(f"\n下一步: python train.py --data_dir {output_base_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  步骤1 - 转换标注（Labelme JSON → 掩码PNG）:")
        print("    python convert_labelme.py convert <json目录> <输出图目录> <输出掩码目录>")
        print()
        print("  步骤2 - 划分数据集（7:1.5:1.5 分为train/val/test）:")
        print("    python convert_labelme.py split <图目录> <掩码目录> <输出根目录>")
        print()
        print("示例:")
        print("  python convert_labelme.py convert ./raw_images ./converted_images ./converted_masks")
        print("  python convert_labelme.py split ./converted_images ./converted_masks ./datasets")
        sys.exit(1)

    command = sys.argv[1]

    if command == "convert":
        # convert命令需要3个参数: json目录、输出图目录、输出掩码目录
        convert_dataset(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "split":
        # split命令需要3个参数: 图目录、掩码目录、输出根目录
        split_dataset(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(f"未知命令: {command}")
        print("可用命令: convert, split")
