import unittest

import numpy as np

from infer_camera_onnx import (
    build_palette,
    build_legend_items,
    count_defect_pixels,
    overlay_mask,
    preprocess_frame,
)


class InferCameraOnnxTest(unittest.TestCase):
    """验证电脑端ONNX摄像头推理脚本中不依赖真实摄像头的核心逻辑。"""

    def test_preprocess_frame_outputs_nchw_float32(self):
        """预处理应把BGR摄像头帧转成ONNX需要的NCHW float32输入。"""
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame_bgr, input_size=224)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

    def test_overlay_mask_keeps_original_frame_shape(self):
        """mask叠加图应保持与原始摄像头画面相同的尺寸和uint8类型。"""
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[20:40, 30:60] = 1

        overlay = overlay_mask(frame_bgr, mask, build_palette(5), alpha=0.45)

        self.assertEqual(overlay.shape, frame_bgr.shape)
        self.assertEqual(overlay.dtype, np.uint8)

    def test_count_defect_pixels_ignores_background(self):
        """缺陷像素统计应只统计非背景类别，背景0不计入NG面积。"""
        mask = np.array(
            [
                [0, 0, 1],
                [2, 0, 4],
            ],
            dtype=np.uint8,
        )

        self.assertEqual(count_defect_pixels(mask), 3)

    def test_build_legend_items_describes_non_background_classes(self):
        """颜色图例应说明非背景类别的ID、显示颜色和当前测试含义。"""
        legend_items = build_legend_items(num_classes=5)

        self.assertEqual(legend_items[0]["class_id"], 1)
        self.assertEqual(legend_items[0]["color_name"], "red")
        self.assertIn("class_1", legend_items[0]["label"])
        self.assertEqual(legend_items[-1]["class_id"], 4)
        self.assertEqual(legend_items[-1]["color_name"], "purple")


if __name__ == "__main__":
    unittest.main()
