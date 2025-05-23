from pai_rag.file.readers.pai.utils.image_utils import is_remote_url


def test_is_remote_url():
    assert is_remote_url("http://www.baidu.com") is True
    assert is_remote_url("https://www.tencent.com/1.jpg") is True
    assert is_remote_url("http://123123213123") is True
    assert is_remote_url("https://abcabc") is True
    assert is_remote_url("/a/b/c/1.jpg") is False
    assert is_remote_url("./1.txt") is False
    assert is_remote_url("../..") is False
    assert is_remote_url(".") is False
    assert is_remote_url("") is False
    assert is_remote_url("/") is False
    assert is_remote_url("data/") is False
