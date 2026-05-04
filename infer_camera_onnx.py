"""
电脑端UVC摄像头ONNX实时推理脚本

用途:
  在模型还没有部署到STM32MP157板子之前，先在电脑上用UVC摄像头跑通完整推理链路。
  这个脚本直接加载export_onnx.py导出的 defect_unet.onnx，用OpenCV读取摄像头画面，
  用onnxruntime执行模型推理，并把分割mask彩色叠加到实时画面上。

推荐运行:
  python infer_camera_onnx.py --camera_id 1

按键:
  q 或 ESC: 退出
  s: 保存当前原图、叠加图、mask图
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# 训练时使用的ImageNet RGB均值和标准差，必须与dataset.py里的get_preprocessing保持一致。
# 如果这里的数值、通道顺序或输入尺寸变了，电脑端/板端推理结果都会和训练验证结果对不上。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Severstal测试数据集是5类：0背景 + 4类钢板缺陷。
# 后续换成你自己的金属零件数据集时，可以保留ID顺序，只改显示名称。
DEFAULT_CLASS_NAMES = [
    "background",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
]


# 颜色名称表与build_palette中的前几种颜色一一对应，用于终端、文档和画面图例。
# OpenCV内部使用BGR数值，但窗口中看到的是正常视觉颜色，这里写的是用户看到的颜色名。
DEFAULT_COLOR_NAMES = [
    "black",
    "red",
    "orange",
    "blue",
    "purple",
    "yellow",
    "green",
]


def get_default_model_path():
    """
    返回默认ONNX模型路径。

    主要流程:
      1. 取当前脚本所在目录，避免用户从别的目录运行时相对路径失效。
      2. 拼出 checkpoints/defect_unet.onnx。

    返回值:
        str: 默认ONNX模型的绝对路径字符串。
    """
    script_dir = Path(__file__).resolve().parent
    return str(script_dir / "checkpoints" / "defect_unet.onnx")


def format_seconds(seconds):
    """
    把秒数格式化为便于终端阅读的字符串。

    参数:
        seconds (float): 耗时秒数。

    返回值:
        str: 形如 12ms、1.23s 的字符串。
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def build_palette(num_classes):
    """
    根据类别数量生成BGR颜色表。

    参数:
        num_classes (int): 模型输出类别数，包含背景类。

    返回值:
        np.ndarray: shape=(num_classes, 3)，dtype=uint8，OpenCV使用的BGR颜色表。

    关键说明:
      - 类别0是背景，颜色固定为黑色，叠加时背景不会被染色。
      - 前几类使用手工挑选的高对比颜色，便于在摄像头画面上快速看出缺陷区域。
      - 如果类别数超过内置颜色数量，就用HSV自动补充颜色，避免索引越界。
    """
    base_palette = np.array(
        [
            [0, 0, 0],        # 0 background: 黑色，占位，不参与背景染色
            [0, 0, 255],      # 1: 红色
            [0, 165, 255],    # 2: 橙色
            [255, 0, 0],      # 3: 蓝色
            [255, 0, 255],    # 4: 紫色
            [0, 255, 255],    # 5: 黄色
            [0, 255, 0],      # 6: 绿色
        ],
        dtype=np.uint8,
    )

    if num_classes <= len(base_palette):
        return base_palette[:num_classes].copy()

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    palette[: len(base_palette)] = base_palette

    # HSV色相均匀取样，给超出内置表的类别生成稳定颜色。
    for class_id in range(len(base_palette), num_classes):
        hue = int(180 * class_id / max(num_classes, 1))
        hsv_color = np.array([[[hue, 220, 255]]], dtype=np.uint8)
        palette[class_id] = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]

    return palette


def build_legend_items(num_classes, class_names=None):
    """
    生成非背景类别的颜色图例数据。

    参数:
        num_classes (int): 模型输出类别数，包含背景类0。
        class_names (list[str] or None): 类别名称列表，None时使用默认class_1/class_2占位名。

    返回值:
        list[dict]: 每个元素描述一个非背景类别，包含类别ID、显示颜色名和图例文字。

    说明:
      当前Severstal测试模型只证明链路可跑通，class_1~class_4不等同于真实零件缺陷名称。
      后续用真实数据训练时，可以把class_names替换成scratch/rust/dent/crack等业务名称。
    """
    names = class_names or DEFAULT_CLASS_NAMES
    legend_items = []

    for class_id in range(1, num_classes):
        if class_id < len(names):
            class_name = names[class_id]
        else:
            class_name = f"class_{class_id}"

        if class_id < len(DEFAULT_COLOR_NAMES):
            color_name = DEFAULT_COLOR_NAMES[class_id]
        else:
            color_name = f"auto_{class_id}"

        legend_items.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "color_name": color_name,
                "label": f"{color_name}=class_{class_id}({class_name})",
            }
        )

    return legend_items


def preprocess_frame(frame_bgr, input_size=224):
    """
    把OpenCV读取到的BGR摄像头帧转换为ONNX模型输入。

    参数:
        frame_bgr (np.ndarray): OpenCV摄像头帧，shape=(H, W, 3)，BGR通道顺序，uint8。
        input_size (int): 模型固定输入尺寸，训练和导出时当前使用224。

    返回值:
        np.ndarray: shape=(1, 3, input_size, input_size)，dtype=float32。

    主要流程:
      1. 检查输入是否是三通道图像。
      2. BGR转RGB，因为训练集读取时是RGB。
      3. resize到训练时固定尺寸224x224。
      4. 像素缩放到0-1，再按ImageNet mean/std归一化。
      5. HWC转NCHW，并增加batch维度。
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr必须是shape=(H, W, 3)的BGR彩色图像")

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    image_float = image_rgb.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD

    input_tensor = np.transpose(image_float, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor.astype(np.float32, copy=False)


def count_defect_pixels(mask):
    """
    统计mask中的缺陷像素数量。

    参数:
        mask (np.ndarray): 单通道类别mask，像素值0表示背景，非0表示缺陷类别。

    返回值:
        int: 非背景像素总数，用于判断OK/NG和在画面上显示缺陷面积。
    """
    return int(np.count_nonzero(mask))


def mask_to_color(mask, palette):
    """
    把单通道类别mask转换成彩色BGR mask。

    参数:
        mask (np.ndarray): 单通道类别mask，shape=(H, W)。
        palette (np.ndarray): BGR颜色表，shape=(num_classes, 3)。

    返回值:
        np.ndarray: 彩色mask，shape=(H, W, 3)，dtype=uint8。

    边界处理:
      如果模型输出了超过颜色表范围的类别ID，会先裁剪到最后一种颜色，
      避免因为异常像素导致程序崩溃。
    """
    safe_mask = np.clip(mask, 0, len(palette) - 1).astype(np.uint8)
    return palette[safe_mask]


def overlay_mask(frame_bgr, mask, palette, alpha=0.45):
    """
    把分割mask叠加到原始摄像头画面上。

    参数:
        frame_bgr (np.ndarray): 原始摄像头帧，shape=(H, W, 3)，BGR格式。
        mask (np.ndarray): 模型输出类别mask，shape通常是(224, 224)。
        palette (np.ndarray): 类别颜色表，BGR格式。
        alpha (float): 缺陷颜色叠加强度，0表示不显示颜色，1表示完全显示颜色。

    返回值:
        np.ndarray: 与原图同尺寸的叠加图，dtype=uint8。

    关键逻辑:
      - mask用最近邻插值放大回摄像头原始尺寸，避免类别ID被线性插值污染。
      - 只对mask非0的缺陷区域做颜色叠加，背景区域保留原始画面。
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
    )

    color_mask = mask_to_color(resized_mask, palette)
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_mask, alpha, 0)
    defect_area = resized_mask > 0

    output = frame_bgr.copy()
    output[defect_area] = blended[defect_area]
    return output


def draw_status_panel(frame_bgr, status, defect_pixels, fps, inference_time, camera_id, provider):
    """
    在画面左上角绘制运行状态。

    参数:
        frame_bgr (np.ndarray): 要写字的画面，函数会原地修改这张图。
        status (str): 当前判定结果，通常是OK或NG。
        defect_pixels (int): 当前帧非背景像素数量。
        fps (float): 实时帧率。
        inference_time (float): 单帧模型推理耗时，单位秒。
        camera_id (int): 当前使用的摄像头编号。
        provider (str): onnxruntime实际使用的执行后端。

    返回值:
        np.ndarray: 写好状态文字的画面，便于链式调用。
    """
    color = (0, 220, 0) if status == "OK" else (0, 0, 255)
    lines = [
        f"{status}  defect_px={defect_pixels}",
        f"FPS={fps:.1f}  infer={format_seconds(inference_time)}",
        f"camera={camera_id}  provider={provider}",
        "q/ESC=quit  s=save",
    ]

    # 先画半透明黑底，保证文字在亮色工件上也能看清。
    panel_h = 26 * len(lines) + 12
    panel_w = 430
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (8, 8), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, dst=frame_bgr)

    for idx, line in enumerate(lines):
        y = 34 + idx * 26
        line_color = color if idx == 0 else (230, 230, 230)
        cv2.putText(
            frame_bgr,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            line_color,
            2,
            cv2.LINE_AA,
        )

    return frame_bgr


def draw_legend(frame_bgr, palette, legend_items):
    """
    在画面底部绘制颜色图例。

    参数:
        frame_bgr (np.ndarray): 要写图例的画面，函数会原地修改。
        palette (np.ndarray): BGR颜色表，类别ID对应颜色。
        legend_items (list[dict]): build_legend_items生成的非背景类别说明。

    返回值:
        np.ndarray: 写好图例的画面。

    关键说明:
      图例只说明“这个颜色对应模型输出的哪个类别ID”。
      当前测试模型没有真实零件业务语义，因此class_1~class_4只是占位类别名。
    """
    if not legend_items:
        return frame_bgr

    frame_h, frame_w = frame_bgr.shape[:2]
    panel_h = 38
    y0 = max(frame_h - panel_h - 8, 0)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (8, y0), (frame_w - 8, frame_h - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, dst=frame_bgr)

    x = 18
    y = y0 + 25
    for item in legend_items:
        class_id = item["class_id"]
        color = tuple(int(v) for v in palette[class_id])
        text = f"{item['color_name']}=class_{class_id}"

        cv2.rectangle(frame_bgr, (x, y - 14), (x + 18, y + 4), color, -1)
        cv2.rectangle(frame_bgr, (x, y - 14), (x + 18, y + 4), (230, 230, 230), 1)
        cv2.putText(
            frame_bgr,
            text,
            (x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        x += 128

        # 如果窗口宽度较窄，就停止绘制后续图例，避免文字挤出窗口。
        if x > frame_w - 120:
            break

    return frame_bgr


def create_session(model_path, provider_mode="auto"):
    """
    创建ONNX Runtime推理会话。

    参数:
        model_path (str): ONNX模型文件路径。
        provider_mode (str): 执行后端选择，auto优先CUDA，cpu强制CPU。

    返回值:
        tuple:
          - session: onnxruntime.InferenceSession对象。
          - provider: 实际使用的执行后端名称。

    说明:
      普通onnxruntime包通常只有CPUExecutionProvider。
      如果以后安装onnxruntime-gpu，并且CUDA可用，auto会优先使用CUDA。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到ONNX模型文件: {model_path}")

    available_providers = ort.get_available_providers()
    if provider_mode == "cpu":
        providers = ["CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    return session, active_provider


def predict_mask(session, input_name, output_name, frame_bgr, input_size):
    """
    对单帧摄像头图像执行ONNX分割推理。

    参数:
        session: onnxruntime.InferenceSession对象。
        input_name (str): ONNX输入节点名称。
        output_name (str): ONNX输出节点名称。
        frame_bgr (np.ndarray): 原始BGR摄像头帧。
        input_size (int): 模型输入尺寸。

    返回值:
        tuple:
          - mask: shape=(input_size, input_size)的类别mask，dtype=uint8。
          - inference_time: 本次session.run耗时，单位秒。
    """
    input_tensor = preprocess_frame(frame_bgr, input_size=input_size)

    start_time = time.perf_counter()
    logits = session.run([output_name], {input_name: input_tensor})[0]
    inference_time = time.perf_counter() - start_time

    if logits.ndim != 4:
        raise RuntimeError(f"ONNX输出维度异常，期望(1,C,H,W)，实际shape={logits.shape}")

    mask = np.argmax(logits, axis=1)[0].astype(np.uint8)
    return mask, inference_time


def save_current_frame(save_dir, frame_bgr, overlay_bgr, mask, palette):
    """
    保存当前帧的原图、叠加图和彩色mask。

    参数:
        save_dir (str): 保存目录。
        frame_bgr (np.ndarray): 原始摄像头帧。
        overlay_bgr (np.ndarray): 缺陷叠加图。
        mask (np.ndarray): 单通道类别mask。
        palette (np.ndarray): 类别颜色表。

    返回值:
        tuple[str, str, str]: 原图、叠加图、mask图的保存路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)
    stem = f"{timestamp}_{milliseconds:03d}"

    raw_path = os.path.join(save_dir, f"{stem}_raw.jpg")
    overlay_path = os.path.join(save_dir, f"{stem}_overlay.jpg")
    mask_path = os.path.join(save_dir, f"{stem}_mask.png")

    color_mask = mask_to_color(mask, palette)
    cv2.imwrite(raw_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(overlay_path, overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(mask_path, color_mask)

    return raw_path, overlay_path, mask_path


def open_camera(camera_id, width, height, fps):
    """
    打开UVC摄像头并设置采集参数。

    参数:
        camera_id (int): 摄像头编号，你当前的UVC摄像头默认是1。
        width (int): 请求采集宽度。
        height (int): 请求采集高度。
        fps (int): 请求采集帧率。

    返回值:
        cv2.VideoCapture: 已打开的摄像头对象。

    说明:
      Windows下使用DirectShow后端通常比默认后端更稳定，尤其是外接UVC摄像头。
      分辨率和帧率只是请求值，实际值以摄像头硬件返回为准。
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {camera_id}。请确认UVC摄像头已连接，且没有被其他软件占用。"
        )

    return cap


def make_display_frame(view_mode, frame_bgr, overlay_bgr, mask, palette):
    """
    根据显示模式生成最终要imshow的画面。

    参数:
        view_mode (str): 显示模式，overlay/side_by_side/mask三选一。
        frame_bgr (np.ndarray): 原始摄像头帧。
        overlay_bgr (np.ndarray): mask叠加图。
        mask (np.ndarray): 单通道类别mask。
        palette (np.ndarray): 类别颜色表。

    返回值:
        np.ndarray: 最终显示画面。
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    if view_mode == "overlay":
        return overlay_bgr

    if view_mode == "mask":
        color_mask = mask_to_color(mask, palette)
        return cv2.resize(color_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    # side_by_side模式用于同时查看原图和叠加结果，便于判断模型是否偏移或误检。
    return np.hstack([frame_bgr, overlay_bgr])


def run_camera_inference(args):
    """
    运行UVC摄像头实时ONNX推理主循环。

    参数:
        args (argparse.Namespace): 命令行参数集合。

    返回值:
        None

    主流程:
      1. 创建ONNX Runtime会话，读取输入/输出节点名称。
      2. 打开UVC摄像头。
      3. 循环读取摄像头帧，执行预处理、推理、mask叠加和状态显示。
      4. 监听键盘，q/ESC退出，s保存当前帧。
    """
    session, provider = create_session(args.model, args.provider)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    palette = build_palette(args.num_classes)
    legend_items = build_legend_items(args.num_classes)

    cap = open_camera(args.camera_id, args.width, args.height, args.fps)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"ONNX模型: {args.model}")
    print(f"ONNX输入: {input_name}, 输出: {output_name}")
    print(f"执行后端: {provider}")
    print(f"摄像头编号: {args.camera_id}")
    print(f"请求分辨率: {args.width}x{args.height} @ {args.fps}fps")
    print(f"实际分辨率: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
    print("按 q 或 ESC 退出，按 s 保存当前帧")

    last_time = time.perf_counter()
    fps_smoothed = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("摄像头读取失败，稍后重试...")
                time.sleep(0.05)
                continue

            mask, inference_time = predict_mask(
                session=session,
                input_name=input_name,
                output_name=output_name,
                frame_bgr=frame_bgr,
                input_size=args.input_size,
            )
            defect_pixels = count_defect_pixels(mask)
            status = "NG" if defect_pixels >= args.threshold_pixels else "OK"
            overlay_bgr = overlay_mask(frame_bgr, mask, palette, alpha=args.alpha)

            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - last_time, 1e-6)
            last_time = current_time
            if fps_smoothed == 0.0:
                fps_smoothed = instant_fps
            else:
                fps_smoothed = fps_smoothed * 0.9 + instant_fps * 0.1

            display_frame = make_display_frame(
                args.view,
                frame_bgr,
                overlay_bgr,
                mask,
                palette,
            )
            draw_status_panel(
                display_frame,
                status=status,
                defect_pixels=defect_pixels,
                fps=fps_smoothed,
                inference_time=inference_time,
                camera_id=args.camera_id,
                provider=provider,
            )
            draw_legend(display_frame, palette, legend_items)

            cv2.imshow(args.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("s"):
                raw_path, overlay_path, mask_path = save_current_frame(
                    args.save_dir,
                    frame_bgr,
                    overlay_bgr,
                    mask,
                    palette,
                )
                print(f"已保存: {raw_path}")
                print(f"已保存: {overlay_path}")
                print(f"已保存: {mask_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    """
    解析命令行参数。

    返回值:
        argparse.Namespace: 参数对象，包含模型路径、摄像头编号、显示阈值等配置。
    """
    parser = argparse.ArgumentParser(
        description="电脑端UVC摄像头ONNX实时缺陷分割推理"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model_path(),
        help="ONNX模型路径，默认使用checkpoints/defect_unet.onnx",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=1,
        help="UVC摄像头编号，当前你的外接UVC摄像头默认是1",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="类别数，Severstal测试模型是5类：背景+4类缺陷",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="模型输入尺寸，必须与训练和ONNX导出保持一致",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="摄像头采集宽度请求值",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="摄像头采集高度请求值",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="摄像头采集帧率请求值",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="缺陷颜色叠加强度，0-1之间",
    )
    parser.add_argument(
        "--threshold_pixels",
        type=int,
        default=80,
        help="判定NG的最小非背景像素数，测试阶段可按误检情况调大或调小",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./camera_infer_outputs",
        help="按s保存截图时的输出目录",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu"],
        default="auto",
        help="ONNX Runtime执行后端，auto优先CUDA，cpu强制CPU",
    )
    parser.add_argument(
        "--view",
        choices=["overlay", "side_by_side", "mask"],
        default="side_by_side",
        help="显示模式：overlay只看叠加图，side_by_side看原图+叠加图，mask只看彩色mask",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default="UVC ONNX Defect Segmentation",
        help="OpenCV显示窗口名称",
    )
    return parser.parse_args()


def main():
    """
    程序入口函数。

    主要流程:
      1. 解析命令行参数。
      2. 检查叠加透明度范围。
      3. 启动摄像头实时推理。
    """
    args = parse_args()
    args.alpha = min(max(args.alpha, 0.0), 1.0)
    run_camera_inference(args)


if __name__ == "__main__":
    main()
