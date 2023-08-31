import requests

class EASAgent:
    def __init__(self, cfg, args):
        self.url = cfg['EASCfg']['url']
        self.token = cfg['EASCfg']['token']

    def post_to_eas(self, query):
        headers = {
            "Authorization": self.token,
            'Accept': "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        query_json = {"prompt": query}
        res = requests.post(
            url=self.url,
            json=query_json,
            # data=query_prompt.encode('utf8'),
            headers=headers,
            timeout=10000,
        )
        return res.text
