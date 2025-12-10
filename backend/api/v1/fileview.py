from fastapi import APIRouter
from pairag.file.store.local_store import SECRET_KEY
import time
import hmac
import hashlib
import base64
from fastapi import Depends, HTTPException, Query
from service.injection import get_tenant_id
from pairag.file.store.file_store_helper import file_store
from fastapi.responses import StreamingResponse

fileview_router = APIRouter()

def verify_signature(file_path: str, expires: int, sig: str) -> bool:
    # 检查时间
    now = int(time.time())
    if now > expires:
        return False

    # 重算签名
    data_to_sign = f"{file_path}:{expires}".encode("utf-8")
    expected = hmac.new(SECRET_KEY, data_to_sign, hashlib.sha256).digest()
    expected_b64 = base64.urlsafe_b64encode(expected).decode("utf-8").rstrip("=")

    # 安全比较
    return hmac.compare_digest(expected_b64, sig)


@fileview_router.get("")
async def fileview(
    file_path: str = Query(...),
    expires: int = Query(...),
    sig: str = Query(...),
    tenant_id: str = Depends(get_tenant_id),
):
    if not verify_signature(file_path, expires, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    file_obj = await file_store.read_async(file_path, tenant_id=tenant_id)
    return StreamingResponse(
        file_obj,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename={file_path}'
        },
    )
