你可以使用命令行向服务侧发送API请求。比如调用[Upload API](#upload-api)上传知识库文件。

## Upload API

支持通过API的方式上传本地文件，并支持指定不同的faiss_path，每次发送API请求会返回一个task_id，之后可以通过task_id来查看文件上传状态（processing、completed、failed）。

- 上传（upload_data）

```bash
curl -X 'POST' http://localhost:8000/api/v1/upload_data -H 'Content-Type: multipart/form-data' -F 'files=@example_data/paul_graham/paul_graham_essay.txt'

# Return: {"task_id": "2c1e557733764fdb9fefa063538914da"}
```

- 查看上传状态（get_upload_state）

```bash
curl 'http://localhost:8000/api/v1/get_upload_state?task_id=1bcea36a1db740d28194df8af40c7226'

# Return: {"task_id":"2c1e557733764fdb9fefa063538914da","status":"completed"}
```

## Query API

- Rag Query请求

```bash
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What did the author do growing up?"}'
```

```bash
# 流式输出
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What did the author do growing up?", "stream":true}'
```

```bash
# 意图识别
curl -X 'POST' http://localhost:8000/service/query -H "Content-Type: application/json" -d '{"question":"现在几点了", "with_intent": true}'
```

- 多轮对话请求

```bash
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What did the author do growing up?"}'
```

```bash
# 传入session_id：对话历史会话唯一标识，传入session_id后，将对话历史进行记录，调用大模型将自动携带存储的对话历史。
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What does he program with?", "session_id": "1702ffxxad3xxx6fxxx97daf7c"}'
```

```bash
# 传入chat_history：用户与模型的对话历史，list中的每个元素是形式为{"user":"用户输入","bot":"模型输出"}的一轮对话，多轮对话按时间顺序排列。
curl -X 'POST' http://localhost:8001/api/v1/query -H "Content-Type: application/json" -d '{"question":"What does he program with?", "chat_history": [{"user":"What did the author do growing up?", "bot":"Growing up, the author worked on writing and programming outside of school. Specifically, he wrote short stories, which he now considers to be awful due to their lack of plot and focus on characters with strong feelings. In terms of programming, he started experimenting with coding in 9th grade using an IBM 1401 at his junior high school, where he and a friend, Rich Draves, got permission to use the machine. They used an early version of Fortran, typing programs on punch cards and running them on the 1401. The experience was limited by the technology, as the only form of input for programs was data stored on punched cards, and the author did not have much data to work with. Later, with the advent of microcomputers, the author''s engagement with programming deepened. He eventually convinced his father to buy a TRS-80, on which he wrote simple games, a program to predict the flight height of model rockets, and even a word processor that his father used to write at least one book."}]}'
```
