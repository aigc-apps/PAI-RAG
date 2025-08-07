# 管理知识库

# 文件管理

- Upload file

```sh
curl -X POST http://localhost:8688/v1/config/knowledgebases/kb21aa6879ce014019b1b91b9ca34bd1d8/files -H 'Content-Type: multipart/form-data' -F 'files=@/Users/feiyue/Documents/test_files/pairag.md;filename=test/pairag.md'
```

```json
{
  "code": 200,
  "message": "文件上传成功",
  "data": [
    {
      "id": "0ea99f76a5fa47beaac51c7e53f3badb",
      "kb_id": "kb21aa6879ce014019b1b91b9ca34bd1d8",
      "message_id": "",
      "file_content": "",
      "file_content_length": 0,
      "file_name": "pairag.md",
      "file_path": "kb21aa6879ce014019b1b91b9ca34bd1d8/docs/test/pairag.md",
      "file_extension": ".md",
      "file_size": 555,
      "file_md5": "c2b99342e160c73f0cf2a2058d031870",
      "file_metadata": {
        "file_path": "kb21aa6879ce014019b1b91b9ca34bd1d8/docs/test/pairag.md",
        "file_name": "pairag.md",
        "file_size": 555,
        "file_extension": ".md"
      },
      "status": "pending",
      "failed_reason": null,
      "active": true,
      "created_at": "2025-08-07T08:17:55.758190",
      "updated_at": "2025-08-07T08:17:55.758206"
    }
  ]
}
```

- Get file by id

```sh
curl -X GET http://localhost:8688/v1/config/knowledgebases/kb21aa6879ce014019b1b91b9ca34bd1d8/files/0ea99f76a5fa47beaac51c7e53f3badb
```

```json
{
  "code": 200,
  "message": "查询知识库文件成功。",
  "data": {
    "id": "0ea99f76a5fa47beaac51c7e53f3badb",
    "kb_id": "kb21aa6879ce014019b1b91b9ca34bd1d8",
    "message_id": "",
    "file_content": "",
    "file_content_length": 0,
    "file_name": "pairag.md",
    "file_path": "kb21aa6879ce014019b1b91b9ca34bd1d8/docs/test/pairag.md",
    "file_extension": ".md",
    "file_size": 555,
    "file_md5": "c2b99342e160c73f0cf2a2058d031870",
    "status": "succeeded",
    "failed_reason": null,
    "active": true,
    "created_at": "2025-08-07T08:17:55.758190",
    "updated_at": "2025-08-07T08:17:55.758206",
    "file_metadata": {
      "file_path": "kb21aa6879ce014019b1b91b9ca34bd1d8/docs/test/pairag.md",
      "file_name": "pairag.md",
      "file_size": 555,
      "file_extension": ".md",
      "file_url": "http://pai-rag.oss-cn-hangzhou.aliyuncs.com/pairag_knowledgebases%2Fkb21aa6879ce014019b1b91b9ca34bd1d8%2Fdocs%2Ftest%2Fpairag.md?OSSAccessKeyId=LTAI5tEdYXHZwuoTgd97KmWX&Expires=1754558373&Signature=JAL5Nk1Lamw56iD3f2l1nDAVdrM%3D"
    }
  }
}
```
