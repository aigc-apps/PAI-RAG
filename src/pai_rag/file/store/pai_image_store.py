import os
from PIL import Image
from io import BytesIO
import math
from PIL.PngImagePlugin import PngImageFile
from loguru import logger

from pai_rag.file.store.oss_store import PaiOssStore

IMAGE_MAX_PIXELS = 512 * 512


class PaiImageStore:
    def __init__(
        self, oss_store: PaiOssStore = None, save_prefix="pai_oss_images/"
    ) -> None:
        self.oss_store = oss_store
        self.save_prefix = save_prefix

    def upload_image(self, image: PngImageFile, doc_name: str):
        if image is None:
            return None

        if self.oss_store is None:
            logger.warning(
                "oss_store is not properly configured, skipping image upload."
            )
            return None

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.width <= 50 or image.height <= 50:
                logger.warning(f"Skipping small image {image}")
                return None

            current_pixels = image.width * image.height

            # 检查像素总数是否超过限制
            if current_pixels > IMAGE_MAX_PIXELS:
                # 计算缩放比例以适应最大像素数
                scale = math.sqrt(IMAGE_MAX_PIXELS / current_pixels)
                new_width = int(image.width * scale)
                new_height = int(image.height * scale)

                # 调整图片大小
                image = image.resize((new_width, new_height), Image.LANCZOS)

            image_stream = BytesIO()
            image.save(image_stream, format="jpeg")

            image_stream.seek(0)
            data = image_stream.getvalue()

            image_url = self.oss_store.put_object_if_not_exists(
                data=data,
                file_ext=".jpeg",
                headers={
                    "x-oss-object-acl": "public-read"
                },  # set public read to make image accessible
                path_prefix=os.path.join(self.save_prefix, doc_name.strip()),
            )
            logger.info(
                f"Saved image {image_url} from {doc_name} with width={image.width}, height={image.height}."
            )
            return image_url
        except Exception as e:
            logger.warning(f"处理图片失败 '{image}': {e}")
