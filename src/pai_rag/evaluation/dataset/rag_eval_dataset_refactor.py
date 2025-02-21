import json
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from pai_rag.evaluation.dataset.rag_qca_dataset_refactor import Source, QcapSample
from loguru import logger


def generate_eval_uuid() -> str:
    return f"eval_{str(uuid.uuid4())}"


class EvalResults(BaseModel):
    results: Dict[str, Any]
    source: Optional[Source] = Source()


class QcapEvalSample(BaseModel):
    """
    主模型，表示完整的 Evaluated QCAP Sample，包含 id, qcap 和 eval_results 等字段。
    """

    id: str = Field(default_factory=generate_eval_uuid)
    qcap: QcapSample
    eval_results: EvalResults


class QcapEvalDataset(BaseModel):
    """
    评估数据集模型，包含多个 QcapEvalSample
    """

    samples: List[QcapEvalSample]

    def save_json(self, file_path: str):
        """
        将数据集保存为 jsonl 文件，每一行是一个 QcapEvalSample 的 JSON 表示。
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for sample in self.samples:
                    f.write(sample.model_dump_json())
                    f.write("\n")
            logger.info(f"数据集已成功保存到 {file_path}")
        except Exception as e:
            logger.info(f"保存数据集时出错: {e}")

    @classmethod
    def from_json(cls, file_path: str) -> "QcapEvalDataset":
        """
        从 jsonl 文件中读取数据，生成一个 QcapEvalDataset 实例。
        """
        samples = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        sample_dict = json.loads(line)
                        sample = QcapEvalSample(**sample_dict)
                        samples.append(sample)
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.info(f"在第 {line_number} 行读取样本时出错: {e}")
            logger.info(f"从 {file_path} 成功加载了 {len(samples)} 个样本")
            return cls(samples=samples)
        except FileNotFoundError:
            logger.info(f"文件 {file_path} 未找到。")
            return cls(samples=[])
        except Exception as e:
            logger.info(f"读取数据集时出错: {e}")
            return cls(samples=[])
