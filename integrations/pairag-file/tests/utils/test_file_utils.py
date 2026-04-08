import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from pairag.file.utils.file_utils import ensure_file_name_is_supported, ensure_file_type_is_supported


class TestEnsureFileNameIsSupported:
    @pytest.mark.parametrize("filename", [
        "doc.pdf", "doc.docx", "doc.xlsx", "doc.csv",
        "doc.txt", "doc.md", "doc.html", "doc.pptx",
        "image.jpg", "image.png", "image.gif",
        "video.mp4", "video.avi",
    ])
    def test_supported_extensions(self, filename):
        ensure_file_name_is_supported(filename)  # Should not raise

    @pytest.mark.parametrize("filename", [
        "malware.exe", "script.bat", "lib.dll", "archive.zip",
        "program.sh", "data.sql",
    ])
    def test_unsupported_extensions_raise(self, filename):
        with pytest.raises(ValueError):
            ensure_file_name_is_supported(filename)

    def test_case_insensitive(self):
        ensure_file_name_is_supported("doc.PDF")  # Should not raise
        ensure_file_name_is_supported("doc.Docx")  # Should not raise


class TestEnsureFileTypeIsSupported:
    def test_supported_extension(self):
        ensure_file_type_is_supported(".pdf")  # Should not raise

    def test_unsupported_extension(self):
        with pytest.raises(ValueError):
            ensure_file_type_is_supported(".exe")

    def test_jsonl_supported(self):
        ensure_file_type_is_supported(".jsonl")  # Should not raise
