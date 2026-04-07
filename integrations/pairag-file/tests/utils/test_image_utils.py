import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from pairag.file.utils.image_utils import (
    is_remote_url,
    markdown_image_text_to_chunk,
    to_markdown_image_text,
)


class TestIsRemoteUrl:
    def test_http_is_remote(self):
        assert is_remote_url("http://example.com/image.png") is True

    def test_https_is_remote(self):
        assert is_remote_url("https://example.com/image.png") is True

    def test_s3_is_remote(self):
        assert is_remote_url("s3://bucket/image.png") is True

    def test_relative_path_not_remote(self):
        assert is_remote_url("./images/photo.png") is False

    def test_absolute_path_not_remote(self):
        assert is_remote_url("/home/user/photo.png") is False

    def test_pathlib_path(self):
        from pathlib import Path
        assert is_remote_url(Path("/home/user/photo.png")) is False


class TestMarkdownImageTextToChunk:
    def test_with_description(self):
        result = markdown_image_text_to_chunk("http://img.png", alt="A photo")
        assert "http://img.png" in result
        assert "A photo" in result
        assert "图片的描述" in result

    def test_empty_description(self):
        result = markdown_image_text_to_chunk("http://img.png", alt="")
        assert "http://img.png" in result
        assert "图片的描述" not in result

    def test_none_description(self):
        result = markdown_image_text_to_chunk("http://img.png", alt=None)
        assert "http://img.png" in result
        assert "图片的描述" not in result


class TestToMarkdownImageText:
    def test_format(self):
        result = to_markdown_image_text("http://img.png", alt="photo")
        assert result == "\n![photo](http://img.png)\n"

    def test_empty_alt(self):
        result = to_markdown_image_text("http://img.png")
        assert result == "\n![](http://img.png)\n"

    def test_contains_url(self):
        url = "https://example.com/test.jpg"
        result = to_markdown_image_text(url, alt="test")
        assert url in result
