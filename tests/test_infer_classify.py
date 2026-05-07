import unittest

import numpy as np

from infer_classify import (
    BAD_CLASS_INDEX,
    GOOD_CLASS_INDEX,
    CLASS_NAMES,
    get_class_probability,
    is_bad_prediction,
    result_name,
    run_inference,
)


class _InputMeta:
    """模拟 ONNX Runtime 输入节点元信息，只提供推理函数需要的 name 字段。"""

    name = "input"


class _FakeSession:
    """模拟 ONNX Runtime 会话，用固定 logits 验证分类索引解释是否正确。"""

    def __init__(self, logits):
        """保存一组固定 logits，run() 被调用时按 ONNX 输出形状返回。"""
        self.logits = np.array([logits], dtype=np.float32)

    def get_inputs(self):
        """返回伪输入节点，供 run_inference() 取得输入名。"""
        return [_InputMeta()]

    def run(self, _output_names, _feed_dict):
        """返回固定 logits，避免测试依赖真实 ONNX 模型文件。"""
        return [self.logits]


class InferClassifyTest(unittest.TestCase):
    """验证分类推理脚本对 ImageFolder 的 bad=0、good=1 映射解释正确。"""

    def test_class_name_order_matches_imagefolder_mapping(self):
        """类别名称顺序必须保持 bad=0、good=1，避免训练和推理解释相反。"""
        self.assertEqual(CLASS_NAMES[BAD_CLASS_INDEX], "bad")
        self.assertEqual(CLASS_NAMES[GOOD_CLASS_INDEX], "good")

    def test_bad_index_zero_is_reported_as_bad(self):
        """当模型 index0 的 logits 更高时，应判为 BAD 且 bad 概率取 probs[0]。"""
        probs, pred_class, _pred_conf = run_inference(
            _FakeSession([5.0, 0.0]),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        )

        self.assertEqual(pred_class, BAD_CLASS_INDEX)
        self.assertTrue(is_bad_prediction(pred_class))
        self.assertEqual(result_name(pred_class), "BAD")
        self.assertGreater(get_class_probability(probs, "bad"), 0.99)
        self.assertLess(get_class_probability(probs, "good"), 0.01)

    def test_good_index_one_is_reported_as_good(self):
        """当模型 index1 的 logits 更高时，应判为 GOOD 且 good 概率取 probs[1]。"""
        probs, pred_class, _pred_conf = run_inference(
            _FakeSession([0.0, 5.0]),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        )

        self.assertEqual(pred_class, GOOD_CLASS_INDEX)
        self.assertFalse(is_bad_prediction(pred_class))
        self.assertEqual(result_name(pred_class), "GOOD")
        self.assertGreater(get_class_probability(probs, "good"), 0.99)
        self.assertLess(get_class_probability(probs, "bad"), 0.01)


if __name__ == "__main__":
    unittest.main()
