from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(5, 15)  # 답변을 받은 뒤 다음 질문까지 대기

    @task
    def chat(self):
        self.client.post(
            "/chat",
            json={"text": "배송 현황을 알려주세요", "dataset": "inquiries"},
            timeout=100,
        )