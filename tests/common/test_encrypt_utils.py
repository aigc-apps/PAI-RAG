import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from common.encrypt_utils import encrypt_key, decrypt_key


class TestEncryptUtils:
    def test_empty_string_returns_as_is(self):
        assert encrypt_key("") == ""
        assert decrypt_key("") == ""

    def test_none_returns_as_is(self):
        assert encrypt_key(None) is None
        assert decrypt_key(None) is None

    def test_roundtrip_consistency(self):
        plaintext = "my-secret-api-key"
        encrypted = encrypt_key(plaintext)
        decrypted = decrypt_key(encrypted)
        assert decrypted == plaintext

    def test_special_characters_roundtrip(self):
        plaintext = "p@$$w0rd!#%^&*()_+={}<>?/中文"
        encrypted = encrypt_key(plaintext)
        decrypted = decrypt_key(encrypted)
        assert decrypted == plaintext

    def test_encrypted_differs_from_plaintext(self):
        plaintext = "my-secret-api-key"
        encrypted = encrypt_key(plaintext)
        assert encrypted != plaintext

    def test_long_string_roundtrip(self):
        plaintext = "x" * 10000
        encrypted = encrypt_key(plaintext)
        decrypted = decrypt_key(encrypted)
        assert decrypted == plaintext
