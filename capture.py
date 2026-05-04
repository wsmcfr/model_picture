"""
UVC摄像头截图脚本
用于在工位上批量采集金属零件图片，供后续Labelme标注和模型训练使用

操作说明:
  - 按 s 键: 截图并保存到指定目录
  - 按 q 键: 退出程序
  - 按 + 键: 增加亮度（模拟不同光照条件采集，增加数据多样性）
  - 按 - 键: 降低亮度
  - 按 0 键: 恢复原始亮度

用法示例:
  拍良品样本: python capture.py --save_dir ./raw_images/good
  拍划痕样本: python capture.py --save_dir ./raw_images/scratch
  拍锈蚀样本: python capture.py --save_dir ./raw_images/rust
  拍压伤样本: python capture.py --save_dir ./raw_images/dent
  拍裂纹样本: python capture.py --save_dir ./raw_images/crack
  拍毛刺样本: python capture.py --save_dir ./raw_images/burr

  指定摄像头编号（默认0，接了多个摄像头时需要试1、2）:
    python capture.py --save_dir ./raw_images/good --camera_id 1

  指定保存质量（默认95，质量越高文件越大但细节保留越好）:
    python capture.py --save_dir ./raw_images/good --quality 98
"""

import cv2
import os
import argparse
import time


def capture(save_dir, camera_id, quality, width, height):
    """
    从UVC摄像头实时截图保存到指定目录

    参数:
        save_dir:   保存目录路径，截图的jpg文件会保存到这里
                    建议按缺陷类型分目录: good/scratch/rust/dent/crack/burr
        camera_id:  摄像头编号，从0开始
                    0 = 系统检测到的第一个摄像头（通常是内置摄像头）
                    1 = 第二个摄像头（外接UVC摄像头）
                    2 = 第三个摄像头
                    如果画面是黑屏或打开的不是目标摄像头，换一个编号试试
        quality:    JPEG保存质量，1-100
                    95 = 推荐，文件大小和画质平衡
                    98 = 高质量，文件更大但保留更多细节，适合小缺陷
                    100= 无损，文件很大
                    低于85不推荐，会丢失细微缺陷纹理
        width:      摄像头采集分辨率宽度（像素）
                    640 = 你的UVC摄像头默认分辨率
                    如果摄像头不支持设定的分辨率，会自动降到支持的分辨率
        height:     摄像头采集分辨率高度（像素）
                    480 = 你的UVC摄像头默认分辨率
    """
    os.makedirs(save_dir, exist_ok=True)

    # 打开摄像头
    # 使用DirectShow后端（cv2.CAP_DSHOW），Windows下对UVC摄像头兼容性更好
    # 默认的Media Foundation后端对部分UVC摄像头会导致卡死
    # cv2.CAP_DSHOW = 700，是Windows DirectShow的API编号
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    # 设置摄像头采集分辨率
    # cap.set 设置的是"请求"分辨率，实际分辨率取决于摄像头硬件支持
    # 你的UVC摄像头是640x480，设定后会匹配到这个分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 读取实际分辨率（摄像头可能不支持设定的值，会自动降到支持的值）
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_id}")
        print("排查方法:")
        print("  1. 检查摄像头是否已连接")
        print("  2. 尝试其他编号: --camera_id 1 或 --camera_id 2")
        print("  3. 检查摄像头是否被其他程序占用")
        return

    print(f"摄像头已打开 (编号={camera_id})")
    print(f"请求分辨率: {width}x{height}")
    print(f"实际分辨率: {actual_width}x{actual_height}")
    if actual_width != width or actual_height != height:
        print(f"注意: 摄像头不支持 {width}x{height}，自动使用了 {actual_width}x{actual_height}")

    # 从已有文件数量继续编号，避免覆盖之前拍的图片
    # 比如目录里已有0000.jpg~0049.jpg，下一张就从0050开始
    existing = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
    count = len(existing)

    # 亮度调整参数（用乘法调整，1.0=原始亮度）
    # 可以在拍摄时用+/-键微调，模拟不同光照条件
    # 这样同一种零件不同亮度拍几张，增加训练数据多样性
    brightness_factor = 1.0

    print(f"\n保存目录: {save_dir}")
    print(f"已有 {count} 张图片，继续从 {count:04d}.jpg 开始编号")
    print(f"JPEG质量: {quality}")
    print(f"\n操作按键:")
    print(f"  s = 截图保存")
    print(f"  q = 退出")
    print(f"  + = 增加亮度（当前: {brightness_factor:.1f}）")
    print(f"  - = 降低亮度（当前: {brightness_factor:.1f}）")
    print(f"  0 = 恢复原始亮度")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败，尝试重新读取...")
            time.sleep(0.1)
            continue

        # 应用亮度调整
        # convertTo 实现像素值缩放: dst = src * alpha + beta
        # alpha > 1 画面变亮, alpha < 1 画面变暗
        display_frame = frame.copy()
        if brightness_factor != 1.0:
            display_frame = display_frame.astype(float)
            display_frame = display_frame * brightness_factor
            display_frame = np.clip(display_frame, 0, 255)
            display_frame = display_frame.astype(np.uint8)

        # 在画面左上角显示状态信息
        # 包括: 已拍数量、亮度系数、操作提示、实际分辨率
        info_text = f"Count:{count} Brightness:{brightness_factor:.1f} Res:{actual_width}x{actual_height}"
        cv2.putText(display_frame, info_text,
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (0, 255, 0), 1)
        cv2.putText(display_frame, "s=save  q=quit  +/-=brightness",
                     (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (0, 200, 0), 1)

        cv2.imshow("Capture - Press s to save, q to quit", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 保存截图
            # 文件名格式: 0000.jpg, 0001.jpg, ..., 9999.jpg
            # 4位数字编号，最多支持10000张
            fname = os.path.join(save_dir, f"{count:04d}.jpg")

            # 保存时应用亮度调整（如果有的话）
            # 这样不同亮度拍的图片可以模拟不同光照条件
            if brightness_factor != 1.0:
                save_frame = frame.astype(float)
                save_frame = save_frame * brightness_factor
                save_frame = np.clip(save_frame, 0, 255)
                save_frame = save_frame.astype(np.uint8)
            else:
                save_frame = frame

            # imwrite 参数:
            #   fname:  文件路径
            #   frame:  图像数据（BGR格式，OpenCV默认）
            #   [cv2.IMWRITE_JPEG_QUALITY, quality]: JPEG质量参数
            #     95 = 推荐，约100-200KB/张（640x480）
            #     100= 无损，约500KB+/张
            cv2.imwrite(fname, save_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            print(f"保存: {fname} (亮度: {brightness_factor:.1f})")
            count += 1

        elif key == ord('q'):
            break

        elif key == ord('+') or key == ord('='):
            # 增加亮度，每次增加0.1，最大3.0（3倍亮度）
            # 用途: 模拟强光照射条件，增加训练数据多样性
            brightness_factor = min(brightness_factor + 0.1, 3.0)
            print(f"亮度调整: {brightness_factor:.1f}")

        elif key == ord('-') or key == ord('_'):
            # 降低亮度，每次减少0.1，最小0.3（0.3倍亮度）
            # 用途: 模拟弱光照射条件，增加训练数据多样性
            brightness_factor = max(brightness_factor - 0.1, 0.3)
            print(f"亮度调整: {brightness_factor:.1f}")

        elif key == ord('0'):
            # 恢复原始亮度（1.0 = 不做任何调整）
            brightness_factor = 1.0
            print(f"亮度恢复: {brightness_factor:.1f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n本次共保存 {count} 张图片到 {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UVC摄像头截图工具 - 用于采集金属零件缺陷检测训练数据"
    )

    # --save_dir: 截图保存目录
    # 必须按缺陷类型分目录保存，方便后续Labelme标注时区分类别
    # 目录名应与Labelme标签名一致: good/scratch/rust/dent/crack/burr
    parser.add_argument(
        "--save_dir", type=str, default="./raw_images",
        help="截图保存目录（建议按缺陷类型分目录: good/scratch/rust/dent/crack/burr）"
    )

    # --camera_id: 摄像头编号
    # Windows下0通常是第一个摄像头，外接UVC摄像头可能是1或2
    # 如果打开的摄像头不对，换编号试试
    parser.add_argument(
        "--camera_id", type=int, default=0,
        help="摄像头编号（0=第一个, 1=第二个, 2=第三个），外接UVC通常为1"
    )

    # --quality: JPEG保存质量
    # 95是推荐值，平衡了文件大小和画质
    # 对于细微缺陷（裂纹、小压伤），建议用98或100保留更多细节
    # 训练时模型需要看到缺陷的纹理细节，质量太低会丢失这些信息
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG保存质量（1-100，默认95，细微缺陷建议98+）"
    )

    # --width / --height: 摄像头采集分辨率
    # 你的UVC摄像头是640x480
    # 如果摄像头不支持设定的分辨率，会自动降到支持的分辨率
    # 采集分辨率越高，缺陷细节越清晰，但640x480对于224x224训练输入已够用
    # 注意: 训练时图片会被缩放到224x224，所以采集分辨率不必太高
    parser.add_argument(
        "--width", type=int, default=640,
        help="摄像头采集宽度（像素，默认640，你的UVC摄像头默认分辨率）"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="摄像头采集高度（像素，默认480，你的UVC摄像头默认分辨率）"
    )

    args = parser.parse_args()

    # 需要numpy来做亮度调整的数值运算
    import numpy as np

    capture(args.save_dir, args.camera_id, args.quality, args.width, args.height)
