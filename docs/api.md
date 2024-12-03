# 🔧 API Service

You can use the command line to send API requests to the server, for example, calling the [Upload Data API](#upload-data-api) to upload a knowledge base file.

## Upload Data API

It supports uploading local files through API and supports specifying different failure_paths. Each time an API request is sent, a task_id will be returned. The file upload status (processing, completed, failed) can then be checked through the task_id.

- upload_data

```bash
curl -X 'POST' http://localhost:8000/api/v1/upload_data -H 'Content-Type: multipart/form-data' -F 'files=@example_data/paul_graham/paul_graham_essay.txt'

# Return: {"task_id": "2c1e557733764fdb9fefa063538914da"}
```

- get_upload_state

```bash
curl 'http://localhost:8000/api/v1/get_upload_state?task_id=1bcea36a1db740d28194df8af40c7226'

# Return: {"task_id":"2c1e557733764fdb9fefa063538914da","status":"completed"}
```

## Query API

- Supports three dialogue modes:
  - /query/retrieval
  - /query/llm
  - /query: (default) RAG (retrieval + llm)

```bash
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What did the author do growing up?"}'
```

```bash
# streaming output
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What did the author do growing up?", "stream":true}'
```

```bash
# with intent
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What's the time", "with_intent":true}'
```

- Multi-round dialogue

```bash
curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What did the author do growing up?"}'
```

> Parameters: session_id
>
> The unique identifier of the conversation history session. After the session_id is passed in, the conversation history will be recorded. Calling the large model will automatically carry the stored conversation history.
>
> ```bash
> curl -X 'POST' http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"question":"What does he program with?", "session_id": "1702ffxxad3xxx6fxxx97daf7c"}'
> ```

> Parameters: chat_history
>
> The conversation history between the user and the model. Each element in the list is a round of conversation in the form of {"user":"user input","bot":"model output"}. Multiple rounds of conversations are arranged in chronological order.
>
> ```bash
> curl -X 'POST' http://localhost:8001/api/v1/query -H "Content-Type: application/json" -d '{"question":"What does he program with?", "chat_history": [{"user":"What did the author do growing up?", "bot":"Growing up, the author worked on writing and programming outside of school. Specifically, he wrote short stories, which he now considers to be awful due to their lack of plot and focus on characters with strong feelings. In terms of programming, he started experimenting with coding in 9th grade using an IBM 1401 at his junior high school, where he and a friend, Rich Draves, got permission to use the machine. They used an early version of Fortran, typing programs on punch cards and running them on the 1401. The experience was limited by the technology, as the only form of input for programs was data stored on punched cards, and the author did not have much data to work with. Later, with the advent of microcomputers, the author''s engagement with programming deepened. He eventually convinced his father to buy a TRS-80, on which he wrote simple games, a program to predict the flight height of model rockets, and even a word processor that his father used to write at least one book."}]}'
> ```
