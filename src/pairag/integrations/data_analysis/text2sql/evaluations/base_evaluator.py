from abc import ABC, abstractmethod


class SQLEvaluator(ABC):
    """生成SQL评估接口"""

    @abstractmethod
    async def abatch_loader(
        self,
    ):
        pass

    @abstractmethod
    async def abatch_query(self, nums: int):
        pass

    @abstractmethod
    def batch_evaluate(
        self, gold_file: str, predicted_file: str, evaluation_type: str, **args
    ):
        pass
