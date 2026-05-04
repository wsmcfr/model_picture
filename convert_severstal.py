"""
Severstal钢材缺陷检测数据集转换脚本

功能：
  将Kaggle Severstal数据集转换为UNet训练管道可用的格式
  直接边转换边保存，不使用临时文件，节省磁盘空间

  步骤：
    1. 读取train.csv中的RLE编码掩码
    2. 先按文件名划分train/val/test（70%/15%/15%）
    3. 对每张图片解码RLE，生成掩码PNG，直接保存到对应split目录
    4. 原图直接复制（shutil.copy2），不经过numpy中转

  注意：Severstal有4类缺陷，训练时用 --num_classes 5

用法：
  python convert_severstal.py
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def rle_decode(rle_string, height, width):
    """
    将RLE编码字符串解码为二值掩码

    Severstal的RLE编码规则：
      - 像素编号从1开始（不是0）
      - 按列优先编号：第1列从上到下，然后第2列，...
      - 格式："起始像素1 长度1 起始像素2 长度2 ..."

    参数：
        rle_string: RLE编码字符串
        height: 图片高度（256）
        width: 图片宽度（1600）

    返回：
        mask: 二值掩码 shape=(height, width), dtype=uint8
    """
    if pd.isna(rle_string) or rle_string.strip() == '':
        return np.zeros((height, width), dtype=np.uint8)

    tokens = rle_string.strip().split()
    numbers = [int(t) for t in tokens]

    mask_flat = np.zeros(height * width, dtype=np.uint8)

    for i in range(0, len(numbers), 2):
        start = numbers[i] - 1
        length = numbers[i + 1]
        mask_flat[start:start + length] = 1

    # 列优先reshape后转置得到行优先的2D掩码
    mask = mask_flat.reshape((width, height)).T
    return mask


def convert_severstal(csv_path, images_dir, output_dir):
    """
    主转换函数

    改进：先划分文件名列表，再逐张转换并直接保存
    不使用临时文件，节省磁盘空间

    参数：
        csv_path:     train.csv路径
        images_dir:   原图目录
        output_dir:   输出目录
    """
    # Severstal图片固定尺寸
    IMG_HEIGHT = 256
    IMG_WIDTH = 1600
    NUM_CLASSES = 5

    # 1. 读取CSV
    print(f"读取CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  记录数: {len(df)}")
    print(f"  类别分布:")
    for cid in sorted(df['ClassId'].unique()):
        count = len(df[df['ClassId'] == cid])
        print(f"    ClassId={cid}: {count}条记录")

    # 2. 获取所有图片文件名
    csv_images = set(df['ImageId'].unique())
    disk_images = sorted(f for f in os.listdir(images_dir) if f.endswith('.jpg'))
    print(f"\n磁盘上总图片数: {len(disk_images)}")
    print(f"CSV中有缺陷的图片: {len(csv_images)}")
    print(f"无缺陷图片: {len(set(disk_images) - csv_images)}")

    # 3. 先划分文件名列表
    all_names = disk_images
    indices = list(range(len(all_names)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42)
    val_ratio = 0.15 / 0.85
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=42)

    split_map = {}
    for i in train_idx:
        split_map[all_names[i]] = 'train'
    for i in val_idx:
        split_map[all_names[i]] = 'val'
    for i in test_idx:
        split_map[all_names[i]] = 'test'

    print(f"\n划分结果:")
    print(f"  train: {len(train_idx)}")
    print(f"  val: {len(val_idx)}")
    print(f"  test: {len(test_idx)}")

    # 4. 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    # 5. 逐张转换并直接保存
    # 预处理：把CSV按ImageId分组，避免每次df[df['ImageId']==name]的O(n)查询
    grouped = df.groupby('ImageId')

    print(f"\n开始转换 {len(all_names)} 张图片...")
    success = 0
    fail = 0

    for idx, img_name in enumerate(all_names):
        try:
            img_path = os.path.join(images_dir, img_name)
            img = Image.open(img_path)
            w, h = img.size
            img_arr = np.array(img)

            # 灰度图转RGB（训练管道需要3通道）
            if len(img_arr.shape) == 2:
                img_arr_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
            else:
                img_arr_rgb = img_arr

            # 创建空掩码
            mask = np.zeros((h, w), dtype=np.uint8)

            # 解码该图片的所有RLE记录
            if img_name in grouped.groups:
                for _, row in grouped.get_group(img_name).iterrows():
                    class_id = row['ClassId']
                    rle = row['EncodedPixels']
                    defect_mask = rle_decode(rle, h, w)
                    mask[defect_mask > 0] = class_id

            # 确定保存到哪个split
            split = split_map[img_name]
            name_no_ext = os.path.splitext(img_name)[0]

            # 保存原图为JPG（直接用PIL保存，不经过npy临时文件）
            img_out_path = os.path.join(
                output_dir, split, 'images', f'{name_no_ext}.jpg')
            Image.fromarray(img_arr_rgb).save(img_out_path, quality=95)

            # 保存掩码为PNG
            mask_out_path = os.path.join(
                output_dir, split, 'masks', f'{name_no_ext}.png')
            Image.fromarray(mask).save(mask_out_path)

            success += 1

            if (idx + 1) % 1000 == 0:
                print(f"  已处理 {idx + 1}/{len(all_names)}")

        except Exception as e:
            print(f"  处理失败: {img_name}, 错误: {e}")
            fail += 1

    print(f"\n转换完成: 成功 {success}, 失败 {fail}")

    # 6. 统计各类别像素占比（抽样）
    print("\n统计各类别像素占比（抽样100张）...")
    for split_name in ['train', 'val', 'test']:
        mask_dir = os.path.join(output_dir, split_name, 'masks')
        if not os.path.isdir(mask_dir):
            continue
        total_pixels = 0
        class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
        fnames = os.listdir(mask_dir)[:100]
        for fname in fnames:
            m = np.array(Image.open(os.path.join(mask_dir, fname)))
            total_pixels += m.size
            for c in range(NUM_CLASSES):
                class_pixels[c] += np.sum(m == c)
        print(f"  {split_name} ({len(fnames)}张):")
        for c in range(NUM_CLASSES):
            pct = class_pixels[c] / total_pixels * 100 if total_pixels > 0 else 0
            print(f"    类别{c}: {pct:.2f}%")

    print(f"\n数据集已生成到: {output_dir}")
    print(f"训练命令:")
    print(f'  D:\\model_picture\\defect-unet\\python.exe train.py --data_dir {output_dir} --num_classes 5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Severstal钢材缺陷数据集转换工具"
    )

    parser.add_argument(
        "--csv", type=str,
        default=r"D:\model_picture\train.csv\train.csv",
        help="train.csv文件路径"
    )
    parser.add_argument(
        "--images", type=str,
        default=r"D:\model_picture\train_images",
        help="训练图片目录路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=r"D:\model_picture\datasets_severstal",
        help="输出目录路径"
    )

    args = parser.parse_args()

    convert_severstal(args.csv, args.images, args.output_dir)
