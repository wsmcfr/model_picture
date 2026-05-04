"""
Severstal数据集分析脚本
用于查看train.csv的格式、类别分布、图片数量等信息
"""
import pandas as pd
import os

csv_path = r"D:\model_picture\train.csv"
img_dir = r"D:\model_picture\train_images"

# 读取CSV
df = pd.read_csv(csv_path)
print(f"CSV shape: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"\n前20行:")
print(df.head(20))
print(f"\n类别分布:")
print(df['ClassId'].value_counts().sort_index())
print(f"\n唯一图片数: {df['ImageId'].nunique()}")
print(f"无缺陷记录数(EncodedPixels为NaN): {df['EncodedPixels'].isna().sum()}")

# 检查图片目录
if os.path.isdir(img_dir):
    imgs = os.listdir(img_dir)
    print(f"\n图片目录中文件数: {len(imgs)}")
else:
    print(f"\n图片目录不存在: {img_dir}")
