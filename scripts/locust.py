from locust import HttpUser, task, between
from random import randint

auth_header = {"Authorization": "<PAI EAS server token>"}

sample_queries = [
    "找一部关于下雪的电影。",
    "找一部适合下雨天看的电影。",
    "心情不好的时候看什么电影？",
    "无聊的时候想看什么电影",
    "压力很大的时候看的电影",
    "好看的中文警匪片",
    "校园爱情电影",
    "金庸小说改编的古装武打剧",
    "好看的仙侠剧",
    "搞笑的电影",
    "评分高的的动画片",
]


class SimpleRagUser(HttpUser):
    wait_time = between(0, 1)

    @task
    def qa(self):
        q_i = randint(0, len(sample_queries) - 1)
        query = sample_queries[q_i]

        _ = self.client.post(
            "/service/query", headers=auth_header, json={"question": query}
        )
        # sprint(response.content.decode("utf-8"))


class SimpleRetrievalUser(HttpUser):
    wait_time = between(0, 1)

    @task
    def qa(self):
        q_i = randint(0, len(sample_queries) - 1)
        query = sample_queries[q_i]

        _ = self.client.post(
            "/service/query/retrieval", headers=auth_header, json={"question": query}
        )
        # print(response.content.decode("utf-8"))
