# evaluator/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEvaluator(ABC):
    """
    评估器基类
    所有评估器必须实现 `evaluate` 方法
    """

    def __init__(self, name: str = "BaseEvaluator"):
        self.name = name

    @abstractmethod
    async def evaluate_async(self, input:str, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        """
        评估预测结果与参考答案的匹配程度

        Args:
            prediction (str): 模型预测输出
            reference (str): 标准答案
            **kwargs: 额外参数（如上下文、评分标准等）

        Returns:
            Dict[str, Any]: 评估结果，必须包含 'score' 字段（0~1）
        """
        pass

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __repr__(self):
        return self.__str__()
