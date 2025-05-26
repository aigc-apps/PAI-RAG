import json
import uuid
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional, Literal, Dict, Union, Any
from loguru import logger


def generate_qca_uuid() -> str:
    return f"qca_{str(uuid.uuid4())}"


def generate_qcap_uuid() -> str:
    return f"qcap_{str(uuid.uuid4())}"


class Source(BaseModel):
    """
    表示来源信息的模型，用于标记模型类型。
    """

    name: Optional[str] = "unknown"
    model: Optional[str] = "unknown"


class Query(BaseModel):
    """
    表示查询信息的模型，包含查询文本和来源信息。
    """

    query_text: str
    source: Optional[Source] = Source()


class TextNodeContext(BaseModel):
    """
    表示类型为 TextNode 的上下文节点。
    """

    node_id: str
    type: Literal["TextNode"]
    text: str
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("metadata")
    def check_metadata_for_text_node(cls, v):
        if "image_url_list" not in v:
            raise ValueError('metadata 必须包含 "image_url_list" 字段')
        if not isinstance(v["image_url_list"], list):
            raise ValueError('"image_url_list" 应该是一个列表')
        return v


class ImageNodeContext(BaseModel):
    """
    表示类型为 ImageNode 的上下文节点。
    """

    node_id: str
    type: Literal["ImageNode"]
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("metadata")
    def check_metadata_for_text_node(cls, v):
        if "image_url" not in v:
            raise ValueError('metadata 必须包含 "image_url" 字段')
        if not isinstance(v["image_url"], str):
            raise ValueError('"image_url" 应该是一个字符串')
        return v


# 定义上下文的联合类型，以支持不同类型的节点
Context = Union[TextNodeContext, ImageNodeContext]


class Answer(BaseModel):
    """
    表示答案信息的模型，包括答案文本、图片 URL 列表和来源信息。
    """

    answer_text: str
    answer_image_url_list: Optional[List[str]] = None
    source: Optional[Source] = Source()


class Prediction(BaseModel):
    """
    表示预测信息的模型，包含预测的contexts和answer。
    """

    contexts: Optional[List[Context]] = None
    answer: Optional[Answer] = None


class QcaSample(BaseModel):
    """
    主模型，表示完整的 QCA Sample，包含 id、query、contexts 和 answer 等字段。
    """

    id: str = Field(default_factory=generate_qca_uuid)
    query: Query
    contexts: Optional[List[Context]] = None
    answer: Optional[Answer] = None

    def get_reference_node_ids(self):
        return [context.node_id for context in self.contexts]

    def get_reference_node_texts(self):
        text_list = []
        for context in self.contexts:
            if type(context) is TextNodeContext:
                text_list.append(context.text)
        return text_list

    def get_reference_image_url_list(self):
        image_url_list = []
        for context in self.contexts:
            if type(context) is TextNodeContext:
                image_url_list.extend(context.metadata["image_url_list"])
            elif type(context) is ImageNodeContext:
                image_url_list.append(context.metadata["image_url"])
        return image_url_list


class QcapSample(BaseModel):
    """
    主模型，表示完整的 QCAP Sample，包含 qca, prediction 和 mode 等字段。
    """

    id: str = Field(default_factory=generate_qcap_uuid)
    qca: QcaSample
    prediction: Prediction
    mode: str

    def get_predicted_node_ids(self):
        return [context.node_id for context in self.prediction.contexts]


class QcaDataset(BaseModel):
    """
    数据集模型，包含多个 QcaSample。
    """

    samples: List[QcaSample]

    def save_json(self, file_path: str):
        """
        将数据集保存为 jsonl 文件，每一行是一个 QcaSample 的 JSON 表示。
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
    def from_json(cls, file_path: str) -> "QcaDataset":
        """
        从 jsonl 文件中读取数据，生成一个 QcaDataset 实例。
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
                        sample = QcaSample(**sample_dict)
                        # sample = QcaSample.model_validate(sample_dict)
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


class QcapDataset(BaseModel):
    """
    数据集模型，包含多个 QcaSample。
    """

    samples: List[QcapSample]

    def save_json(self, file_path: str):
        """
        将数据集保存为 jsonl 文件，每一行是一个 QcapSample 的 JSON 表示。
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
    def from_json(cls, file_path: str) -> "QcapDataset":
        """
        从 jsonl 文件中读取数据，生成一个 QcapDataset 实例。
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
                        sample = QcapSample(**sample_dict)
                        # sample = QcapSample.model_validate(sample_dict)
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
