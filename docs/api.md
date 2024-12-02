# 🔧 API Service

You can use the command line to send API requests to the server, for example, calling the [Upload Data API](#upload-data-api) to upload a knowledge base file.

## Upload Data API

It supports uploading local files through API and supports specifying different failure_paths. Each time an API request is sent, a task_id will be returned. The file upload status (processing, completed, failed) can then be checked through the task_id.

- upload_data

```bash
curl -X 'POST' http://127.0.0.1:8000/service/upload_data -H 'Content-Type: multipart/form-data' -F 'files=@local_path/PAI.txt'

# Return: {"task_id": "2c1e557733764fdb9fefa063538914da"}
```

- get_upload_state

```bash
curl http://127.0.0.1:8001/service/get_upload_state\?task_id\=2c1e557733764fdb9fefa063538914da

# Return: {"task_id":"2c1e557733764fdb9fefa063538914da","status":"completed"}
```

## Query API

- Supports three dialogue modes:
  - /query/retrieval
  - /query/llm
  - /query: (default) RAG (retrieval + llm)

```bash
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"PAI是什么？"}'
```

```bash
# streaming output
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"PAI是什么？", "stream":true}'
```

```bash
# with intent
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"现在几点了", "with_intent":true}'
```

- Multi-round dialogue

```bash
curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"What is PAI?"}'
```

> Parameters: session_id
>
> The unique identifier of the conversation history session. After the session_id is passed in, the conversation history will be recorded. Calling the large model will automatically carry the stored conversation history.
>
> ```bash
> curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"What are its advantages?", "session_id": "1702ffxxad3xxx6fxxx97daf7c"}'
> ```

> Parameters: chat_history
>
> The conversation history between the user and the model. Each element in the list is a round of conversation in the form of {"user":"user input","bot":"model output"}. Multiple rounds of conversations are arranged in chronological order.
>
> ```bash
> curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"What are its features？", "chat_history": [{"user":"What is PAI?", "bot":"PAI is Alibaba Cloud's artificial intelligence platform, which provides a one-stop machine learning solution. This platform supports various machine learning tasks, including supervised learning, unsupervised learning, and reinforcement learning, and is suitable for multiple scenarios such as marketing, finance, and social networks."}]}'
> ```

> Parameters: session_id + chat_history
>
> Chat_history will be used to append and update the conversation history corresponding to the stored session_id
>
> ```bash
> curl -X 'POST' http://127.0.0.1:8000/service/query -H "Content-Type: application/json" -d '{"question":"What are its advantages?", "chat_history": [{"user":"PAI是什么？", "bot":"PAI is Alibaba Cloud's artificial intelligence platform, which provides a one-stop machine learning solution. This platform supports various machine learning tasks, including supervised learning, unsupervised learning, and reinforcement learning, and is suitable for multiple scenarios such as marketing, finance, and social networks."}], "session_id": "1702ffxxad3xxx6fxxx97daf7c"}'
> ```
