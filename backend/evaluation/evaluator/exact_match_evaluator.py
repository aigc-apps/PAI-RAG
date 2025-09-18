# evaluator/exact_match_evaluator.py
import re
from typing import Dict, Any
from .base import BaseEvaluator

class ExactMatchEvaluator(BaseEvaluator):
    """
    精确匹配评估器（忽略标点、空格、大小写）
    适用于答案标准化、选择题、填空题等场景
    """

    def __init__(self, name: str = "ExactMatch", normalize_spaces: bool = True, ignore_punctuation: bool = True, case_sensitive: bool = False):
        super().__init__(name)
        self.normalize_spaces = normalize_spaces
        self.ignore_punctuation = ignore_punctuation
        self.case_sensitive = case_sensitive

    def _normalize(self, text: str) -> str:
        """标准化文本：去标点、去多余空格、转小写"""
        if not self.case_sensitive:
            text = text.lower()

        if self.ignore_punctuation:
            # 移除所有标点符号（保留中文、字母、数字、空格）
            text = re.sub(r'[^\w\s]', '', text)

        if self.normalize_spaces:
            # 合并多个空格为一个，去除首尾空格
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def evaluate_async(self, input:str, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        pred_norm = self._normalize(prediction)
        ref_norm = self._normalize(reference)

        score = 1.0 if pred_norm == ref_norm else 0.0
        reason = "预测答案与参考答案一致" if pred_norm == ref_norm else "预测答案与参考答案不一致"

        return {
            "score": score,
            "prediction_normalized": pred_norm,
            "reference_normalized": ref_norm,
            "reason": reason,
            "is_match": score == 1.0,
            "evaluator": self.name,
        }
