"""
MobileNetV3-Small 分类模型本地推理脚本

用途:
  在 PC 端测试分类模型效果，支持单张图片推理和摄像头实时推理
  加载 export_classify_onnx.py 导出的 ONNX 模型，输出 good/bad 二分类结果

输入规格:
  与训练一致: (1, 3, 224, 224) RGB, ImageNet 归一化
  预处理: Resize(224,224) -> ToTensor -> Normalize(mean, std)

用法:
  单张图片推理:
    D:\model_picture\defect-unet\python.exe infer_classify.py --mode image --input ./test.jpg --model ./checkpoints_classify/defect_classifier.onnx

  摄像头实时推理:
    D:\model_picture\defect-unet\python.exe infer_classify.py --mode camera --camera_id 0 --model ./checkpoints_classify/defect_classifier.onnx

  批量图片推理:
    D:\model_picture\defect-unet\python.exe infer_classify.py --mode batch --input ./test_images/ --model ./checkpoints_classify/defect_classifier.onnx
"""

import argparse
import os
import time
import glob

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# 全局常量：必须与训练时完全一致
# ============================================================

# ImageNet 归一化参数，与 train_classify.py / export_classify_onnx.py 保持一致
# mean 和 std 顺序都是 RGB（OpenCV 默认读取 BGR，后面会手动转通道）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 输入尺寸，与训练时 Resize((224, 224)) 一致
INPUT_SIZE = (224, 224)

# 类别名称，索引 0=good, 1=bad，与 train_classify.py 的 class_to_idx 对应
CLASS_NAMES = ["good", "bad"]

# 类别颜色（BGR 格式，用于 OpenCV 显示）
CLASS_COLORS = {
    "good": (0, 255, 0),    # 绿色
    "bad":  (0, 0, 255),    # 红色
}

# 置信度阈值：bad 概率超过此值判定为缺陷品
BAD_THRESHOLD = 0.5


def preprocess_image(image_bgr):
    """
    对单张 BGR 图片做分类预处理，输出模型可直接喂入的 numpy 数组

    参数:
        image_bgr (np.ndarray): OpenCV 读取的原始图片，形状 (H, W, 3)，BGR 格式，uint8

    返回:
        input_tensor (np.ndarray): 预处理后的输入，形状 (1, 3, 224, 224)，float32
        display_img (np.ndarray):  缩放后的 224x224 BGR 图，仅用于画面显示

    处理流程:
        1. 直接 Resize 到 224x224（与训练一致，不保持比例，零件占满输入）
        2. BGR -> RGB 通道转换
        3. 归一化到 [0,1] 然后做 ImageNet 标准化: (x/255 - mean) / std
        4. HWC -> CHW 维度变换
        5. 增加 batch 维度: (1, 3, 224, 224)
    """
    # 步骤1: 直接缩放到 224x224
    # 与训练时的 transforms.Resize((224, 224)) 保持一致
    # 分类任务不需要保持比例，Resize 保证零件占满输入区域
    resized = cv2.resize(image_bgr, INPUT_SIZE)

    # 步骤2: BGR -> RGB
    # OpenCV 默认读取的是 BGR，而训练时 PIL 读取的是 RGB
    # 必须转换，否则 R 和 B 通道反了，预训练权重不匹配
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 步骤3: 归一化
    # 先转 float32，然后除以 255 映射到 [0,1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # ImageNet 标准化: (pixel - mean) / std
    # 对 R/G/B 三个通道分别计算
    image_norm = (image_float - IMAGENET_MEAN) / IMAGENET_STD

    # 步骤4: HWC -> CHW
    # numpy 数组形状从 (224, 224, 3) 变为 (3, 224, 224)
    image_chw = np.transpose(image_norm, (2, 0, 1))

    # 步骤5: 增加 batch 维度
    # 最终形状: (1, 3, 224, 224)，与 ONNX 输入签名一致
    input_tensor = np.expand_dims(image_chw, axis=0).astype(np.float32)

    return input_tensor, resized


def run_inference(session, input_tensor):
    """
    执行 ONNX 模型推理

    参数:
        session (ort.InferenceSession): onnxruntime 推理会话
        input_tensor (np.ndarray): 预处理后的输入，形状 (1, 3, 224, 224)

    返回:
        probs (np.ndarray): Softmax 后的概率，形状 (2,)，[good_prob, bad_prob]
        pred_class (int): 预测类别索引，0=good, 1=bad
        pred_conf (float): 预测类别的置信度
    """
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name

    # 执行推理
    # outputs 是列表，这里只有一个输出 "output"，形状 (1, 2)
    outputs = session.run(None, {input_name: input_tensor})

    # 取第一个输出，去掉 batch 维度，得到 (2,) 的 logits
    logits = outputs[0][0]

    # Softmax 转为概率
    # exp(x) / sum(exp(x))，数值稳定性处理：先减最大值
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # 预测类别和置信度
    pred_class = int(np.argmax(probs))
    pred_conf = float(probs[pred_class])

    return probs, pred_class, pred_conf


def draw_result(image, probs, pred_class, pred_conf, inference_time_ms):
    """
    在图片上绘制分类结果文字

    参数:
        image (np.ndarray): 要绘制的 BGR 图片，会被原地修改
        probs (np.ndarray): 两个类别的概率 [good_prob, bad_prob]
        pred_class (int): 预测类别索引
        pred_conf (float): 置信度
        inference_time_ms (float): 推理耗时（毫秒）

    返回:
        image (np.ndarray): 绘制后的图片
    """
    h, w = image.shape[:2]

    # 判定结果文字
    result_text = f"{CLASS_NAMES[pred_class].upper()}: {pred_conf*100:.1f}%"
    bad_prob = probs[1]

    # 颜色：bad 用红色，good 用绿色
    color = CLASS_COLORS["bad"] if pred_class == 1 else CLASS_COLORS["good"]

    # 背景条：顶部画一条色带直观显示结果
    bar_height = int(h * 0.08)
    cv2.rectangle(image, (0, 0), (w, bar_height), color, -1)

    # 在色带上写结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 500.0)
    thickness = max(1, int(w / 250))
    text_y = int(bar_height * 0.7)
    cv2.putText(image, result_text, (10, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 左下角写详细概率和耗时
    info_lines = [
        f"Good: {probs[0]*100:.1f}%",
        f"Bad:  {probs[1]*100:.1f}%",
        f"Time: {inference_time_ms:.1f}ms",
    ]

    line_height = int(h * 0.05)
    for i, line in enumerate(info_lines):
        y = h - 10 - (len(info_lines) - 1 - i) * line_height
        cv2.putText(image, line, (10, y), font, font_scale * 0.7, (255, 255, 255), thickness, cv2.LINE_AA)

    # 如果判定为 bad，加红色边框告警
    if pred_class == 1:
        border = int(max(3, w * 0.015))
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), CLASS_COLORS["bad"], border)

    return image


def infer_single_image(session, image_path, save_dir=None):
    """
    对单张图片进行推理并显示/保存结果

    参数:
        session (ort.InferenceSession): ONNX 推理会话
        image_path (str): 图片文件路径
        save_dir (str or None): 结果保存目录，None 则只显示不保存

    返回:
        pred_class (int): 预测类别
        probs (np.ndarray): 概率分布
    """
    # 读取图片
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return None, None

    # 预处理
    input_tensor, resized = preprocess_image(image_bgr)

    # 推理
    t0 = time.perf_counter()
    probs, pred_class, pred_conf = run_inference(session, input_tensor)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    # 判定阈值逻辑
    # 即使 softmax 输出 bad 概率接近 0.5，也按 argmax 结果走
    # 用户可以通过 BAD_THRESHOLD 调整敏感度
    result_str = "BAD" if pred_class == 1 else "GOOD"
    print(f"[结果] {os.path.basename(image_path)} -> {result_str} "
          f"(good={probs[0]*100:.2f}%, bad={probs[1]*100:.2f}%, time={infer_ms:.1f}ms)")

    # 绘制结果画面（基于 resized 224x224 图）
    display = draw_result(resized.copy(), probs, pred_class, pred_conf, infer_ms)

    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_{result_str.lower()}.jpg")
        cv2.imwrite(save_path, display)
        print(f"[保存] {save_path}")

    # 显示结果窗口
    win_name = "Classification Result"
    cv2.imshow(win_name, display)
    print("[提示] 按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)

    return pred_class, probs


def infer_camera(session, camera_id=0, save_dir=None):
    """
    摄像头实时分类推理

    参数:
        session (ort.InferenceSession): ONNX 推理会话
        camera_id (int): 摄像头设备编号，0=默认摄像头
        save_dir (str or None): 按 s 键保存截图的目录

    说明:
        实时读取摄像头画面，每帧都做 Resize(224,224) + 推理 + 结果显示
        性能取决于 PC 性能，一般 30fps 以上很轻松
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头: {camera_id}")
        return

    # 尝试设置分辨率为 640x480（与用户相机一致）
    # 如果摄像头不支持，会自动 fallback 到最近可用分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[摄像头] 已开启，实际分辨率: {actual_w}x{actual_h}")
    print("[操作] 按 'q' 或 ESC 退出，按 's' 保存当前帧")

    frame_count = 0
    total_infer_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 摄像头读取失败")
            break

        # 预处理 + 推理
        input_tensor, _ = preprocess_image(frame)

        t0 = time.perf_counter()
        probs, pred_class, pred_conf = run_inference(session, input_tensor)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        frame_count += 1
        total_infer_ms += infer_ms

        # 在原图上绘制结果（用一个小窗口叠加）
        result_str = "BAD" if pred_class == 1 else "GOOD"
        color = CLASS_COLORS["bad"] if pred_class == 1 else CLASS_COLORS["good"]

        # 左上角信息面板
        info_text = f"{result_str} G:{probs[0]*100:.0f}% B:{probs[1]*100:.0f}% {infer_ms:.1f}ms"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # 如果 bad，画红色边框
        if pred_class == 1:
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 3)

        # 显示 FPS（过去 30 帧平均）
        avg_ms = total_infer_ms / frame_count
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        fps_text = f"FPS: {fps:.1f} (avg {avg_ms:.1f}ms)"
        cv2.putText(frame, fps_text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Camera Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s') and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"capture_{ts}_{result_str.lower()}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"[保存] {save_path}")

    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_ms = total_infer_ms / frame_count
        print(f"[统计] 共推理 {frame_count} 帧，平均每帧 {avg_ms:.2f}ms ({1000.0/avg_ms:.1f} FPS)")


def infer_batch(session, input_dir, save_dir=None):
    """
    批量推理目录下的所有图片

    参数:
        session (ort.InferenceSession): ONNX 推理会话
        input_dir (str): 图片目录路径
        save_dir (str or None): 结果保存目录

    说明:
        遍历目录下所有 .jpg/.jpeg/.png/.bmp 文件，逐个推理并打印结果
        最后输出统计信息：good 数量、bad 数量、平均耗时
    """
    # 收集所有图片文件
    patterns = [os.path.join(input_dir, "*." + ext) for ext in ["jpg", "jpeg", "png", "bmp"]]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))

    if not image_paths:
        print(f"[错误] 目录下未找到图片: {input_dir}")
        return

    print(f"[批量推理] 共找到 {len(image_paths)} 张图片")

    results = {
        "good": 0,
        "bad": 0,
        "total_ms": 0.0,
    }

    for img_path in sorted(image_paths):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        input_tensor, resized = preprocess_image(image_bgr)

        t0 = time.perf_counter()
        probs, pred_class, pred_conf = run_inference(session, input_tensor)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        results["total_ms"] += infer_ms
        class_name = CLASS_NAMES[pred_class]
        results[class_name] += 1

        bad_prob = probs[1]
        print(f"  {os.path.basename(img_path):30s} -> {class_name.upper():4s} "
              f"(bad_prob={bad_prob*100:6.2f}%, time={infer_ms:5.1f}ms)")

        # 保存带标注的结果图
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            display = draw_result(resized.copy(), probs, pred_class, pred_conf, infer_ms)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(save_dir, f"{base}_{class_name}.jpg")
            cv2.imwrite(out_path, display)

    # 统计汇总
    total = results["good"] + results["bad"]
    avg_ms = results["total_ms"] / total if total > 0 else 0
    print(f"\n[汇总] 总图片数: {total}, Good: {results['good']}, Bad: {results['bad']}")
    print(f"       平均推理耗时: {avg_ms:.2f}ms, 预估 FPS: {1000.0/avg_ms:.1f}")


def main():
    """
    主入口：解析命令行参数，加载 ONNX 模型，根据模式执行推理
    """
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small 分类模型 ONNX 推理"
    )

    # ---- 模型参数 ----
    parser.add_argument(
        "--model", type=str, default="./checkpoints_classify/defect_classifier.onnx",
        help="ONNX 模型文件路径，默认 ./checkpoints_classify/defect_classifier.onnx"
    )

    # ---- 运行模式 ----
    parser.add_argument(
        "--mode", type=str, default="image", choices=["image", "camera", "batch"],
        help="推理模式: image=单张图片, camera=摄像头实时, batch=批量目录"
    )

    # ---- 输入参数 ----
    parser.add_argument(
        "--input", type=str, default="",
        help="输入图片路径（mode=image）或输入目录（mode=batch）"
    )
    parser.add_argument(
        "--camera_id", type=int, default=0,
        help="摄像头设备编号（mode=camera），0=默认摄像头，外接UVC通常是1或2"
    )

    # ---- 输出参数 ----
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="结果保存目录，默认 None（只显示不保存）"
    )

    # ---- 推理后端 ----
    parser.add_argument(
        "--provider", type=str, default="auto",
        help="ONNXRuntime 执行提供器: auto/CUDAExecutionProvider/CPUExecutionProvider"
    )

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"找不到 ONNX 模型: {args.model}\n"
            f"请先运行 export_classify_onnx.py 导出模型。"
        )

    # 选择执行提供器
    if args.provider == "auto":
        # 优先使用 CUDA（NVIDIA 显卡），否则 fallback 到 CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = [args.provider]

    # 加载 ONNX 模型
    print(f"[加载模型] {args.model}")
    session = ort.InferenceSession(args.model, providers=providers)

    # 打印模型信息
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(f"  输入: {input_meta.name}, shape={input_meta.shape}, dtype={input_meta.type}")
    print(f"  输出: {output_meta.name}, shape={output_meta.shape}, dtype={output_meta.type}")
    print(f"  后端: {session.get_providers()}")

    # 根据模式执行推理
    if args.mode == "image":
        if not args.input:
            raise ValueError("mode=image 时必须指定 --input 图片路径")
        infer_single_image(session, args.input, args.save_dir)

    elif args.mode == "camera":
        infer_camera(session, args.camera_id, args.save_dir)

    elif args.mode == "batch":
        if not args.input:
            raise ValueError("mode=batch 时必须指定 --input 图片目录路径")
        infer_batch(session, args.input, args.save_dir)

    print("[完成] 推理结束")


if __name__ == "__main__":
    main()
