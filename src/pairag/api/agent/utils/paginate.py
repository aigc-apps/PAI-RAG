from pydantic import BaseModel

# 分页信息模型
class PaginationMeta(BaseModel):
    total: int
    page: int
    offset: int
    size: int
    total_pages: int

def get_pagination_meta(page:int, size:int, total: int) -> PaginationMeta:
    # 计算 offset
    offset = (page - 1) * size

    # 计算分页信息
    total_pages = (total + size - 1) // size

    # 构造分页信息
    pagination = PaginationMeta(
        total=total,
        page=page,
        size=size,
        offset=offset,
        total_pages=total_pages
    )

    return pagination
