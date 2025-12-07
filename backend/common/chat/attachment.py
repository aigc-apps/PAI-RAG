from pydantic import BaseModel, Field
from typing import List


class Attachment(BaseModel):
    file_id: str
    file_name: str
    file_content: str
    file_extension: str
    file_path: str


class AttachmentCollection(BaseModel):
    code_files: List[Attachment] = Field(default=[])
    image_files: List[Attachment] = Field(default=[])
    text_files: List[Attachment] = Field(default=[])
