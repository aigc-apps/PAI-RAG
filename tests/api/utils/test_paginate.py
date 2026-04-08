import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

from api.v1.utils.paginate import get_pagination_meta, PaginationMeta


class TestGetPaginationMeta:
    def test_first_page(self):
        meta = get_pagination_meta(page=1, size=10, total=25)
        assert meta.total == 25
        assert meta.pages == 3
        assert meta.page == 1
        assert meta.size == 10
        assert meta.offset == 0

    def test_middle_page(self):
        meta = get_pagination_meta(page=2, size=10, total=25)
        assert meta.offset == 10
        assert meta.page == 2

    def test_last_page(self):
        meta = get_pagination_meta(page=3, size=10, total=25)
        assert meta.offset == 20
        assert meta.pages == 3

    def test_exact_division(self):
        meta = get_pagination_meta(page=1, size=10, total=20)
        assert meta.pages == 2

    def test_single_item(self):
        meta = get_pagination_meta(page=1, size=10, total=1)
        assert meta.pages == 1
        assert meta.offset == 0

    def test_zero_total(self):
        meta = get_pagination_meta(page=1, size=10, total=0)
        assert meta.pages == 0
        assert meta.total == 0

    def test_returns_pagination_meta_model(self):
        meta = get_pagination_meta(page=1, size=5, total=50)
        assert isinstance(meta, PaginationMeta)
