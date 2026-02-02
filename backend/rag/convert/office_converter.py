import subprocess
import os
import tempfile
from io import BytesIO
from typing import BinaryIO
from loguru import logger
import pandas as pd
import traceback

def convert_xls_to_xlsx(input_file: BinaryIO):
    try:
        output_file = BytesIO()
        df = pd.read_excel(input_file, sheet_name=0, engine='xlrd')
        df.to_excel(output_file, engine='openpyxl', index=False)
        output_file.seek(0)
        logger.info("Successfully converted to xlsx")
        return output_file
    except Exception as e:
        logger.error(f"❌ Excel file conversion failed: {traceback.format_exc()}")
        raise Exception(f"Failed to convert to xlsx: {e}")


def convert_doc_to_docx(input_file: BinaryIO) -> BinaryIO:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.doc")
            output_path = os.path.join(temp_dir, "input.docx")
            with open(input_path, "wb") as f:
                f.write(input_file.read())
                subprocess.run(["libreoffice", "--headless", "--convert-to", "docx", input_path, "--outdir", temp_dir], check=True)
                with open(output_path, "rb") as f:
                    data = f.read()
                output_file = BytesIO(data)
                logger.info("Successfully converted to docx")
                return output_file
    except Exception as e:
        logger.error(f"Failed to convert to docx: {traceback.format_exc()}")
        raise Exception(f"Failed to convert to docx: {e}")


def convert_ppt_to_pptx(input_file: BinaryIO) -> BinaryIO:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.ppt")
            output_path = os.path.join(temp_dir, "input.pptx")
            with open(input_path, "wb") as f:
                f.write(input_file.read())
                subprocess.run(["libreoffice", "--headless", "--convert-to", "pptx", input_path, "--outdir", temp_dir], check=True)
                with open(output_path, "rb") as f:
                    data = f.read()
                output_file = BytesIO(data)
                logger.info("Successfully converted to pptx")
                return output_file
    except Exception as e:
        logger.error(f"Failed to convert to pptx: {traceback.format_exc()}")
        raise Exception(f"Failed to convert to pptx: {e}")
