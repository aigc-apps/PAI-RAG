# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

import requests
import json

class EASAgent:
    def __init__(self, cfg):
        self.url = cfg['EASCfg']['url']
        self.token = cfg['EASCfg']['token']

    def post_to_eas(self, query):
        headers = {
            "Authorization": self.token,
            'Accept': "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        # query_json = {"prompt": query}
        res = requests.post(
            url=self.url,
            # json=query_json,
            data=query.encode('utf8'),
            headers=headers,
            timeout=10000,
        )
        # return json.loads(res.text)['response']
        return res.text
