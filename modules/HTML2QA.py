import requests
import re
import time
import hashlib
from requests.exceptions import SSLError
from utils.filter import fliter
from utils.splitter import spliter
from utils.generator import HtmlGenerator
from loguru import logger
class HTML2QA:
    def __init__(self, config):
        self.config = config['HTMLCfg']
        self.genertor = HtmlGenerator(self.config)
    
    def deal_Q(self, question, theme, hn, answer, history_QA_dict):
        if not hn in question:
            Q_save = hn + " " + question
        else:
            Q_save = question
        if theme not in question:
            Q_save = theme + " " + Q_save
        else:
            Q_save = Q_save
        if Q_save not in history_QA_dict or len(history_QA_dict[Q_save]) < len(answer):
            history_QA_dict[Q_save] = answer
            return True
        return False

    def check_answer(self, answer):
        ban_words=("是什么", "正文中没有", "没有在正文部分提及", "访问错误中心", "参考相关文档", "抱歉", "无法回答")
        for bw in ban_words:
            if bw in answer:
                return False
        return True

    def check_question(self, question):
        ban_words = ("<h3>", )
        for bw in ban_words:
            if bw in question:
                return False
        return True

    def fliter_html_label_all(self, text):
        max_len = 999
        code_start = [obj.span()[0] for obj in re.finditer("<code", text)]
        code_end = [obj.span()[1] for obj in re.finditer("</code>", text)]
        for i in range(len(code_end)-1, -1, -1):
            text = text[:code_start[i]] + text[code_end[i]:]    # 删除 <code></code> 元素
        pattern = "<[^>]{1,%d}>" % max_len
        html_label_pattern = re.compile(pattern)
        find_iter = list(html_label_pattern.finditer(text))
        for f_i in range(len(find_iter)-1, -1, -1):
            f_it = find_iter[f_i]
            l_p, r_p = f_it.span()[0], f_it.span()[1]
            text = text[:l_p] + text[r_p:]  # 删除所有 <...> 标签
        return text

    def make_url(self, node_id):
        base_url = "https://help.aliyun.com/document_detail/"
        return "{}{}.html".format(base_url, node_id)

    def make_MD5(self, text):
        hl = hashlib.md5()
        hl.update(text.encode(encoding='utf-8'))
        return hl.hexdigest()

    def check_url(self, url, ban_urls=(), ban_keys=()):
        for ban_key in ban_keys:
            if ban_key in url:
                return False
        if url in ban_urls:
            return False
        else:
            return True
    
    def get_html_code(self, url):
        try:
            time.sleep(0.02)
            response = requests.get(url)  # 发送请求，获取响应
            if response.status_code == 200:  # 状态码为200表示请求成功
                html = response.text  # 获取网页源码
                return html  # 输出网页源码
            else:
                raise RuntimeError("[Request Error]网页请求失败，状态码：", response.status_code)
        except SSLError as ssl:
            # 换成http
            time.sleep(0.02)
            url = url.replace("https", "http")
            response = requests.get(url)  # 发送请求，获取响应
            if response.status_code == 200:  # 状态码为200表示请求成功
                html = response.text  # 获取网页源码
                return html  # 输出网页源码
            else:
                raise RuntimeError("[Request Error]网页请求失败，状态码：", response.status_code)

    def loader(self, url):
        if self.check_url(url):
            return self.get_html_code(url)
        else:
            raise RuntimeError("[Bad Url]当前url不是正确的url")
    
    def check_sub_doc(self, i, sub_doc):
        ban_patterns = (
            "<h2>[^<]*附录[^<]*</h2>",
            "<h2>[^<]*联系我们[^<]*</h2>",
            "<h2>[^<]*示例数据[^<]*</h2>",
            "<video[^<]*</video>"
        )
        for bp in ban_patterns:
            if re.search(bp, sub_doc):
                message = "[Bad Context]html的Context中存在需要过滤的内容 ban:%s \nsub_doc:%s" % (bp, sub_doc)
                return message
        if re.finditer("示例", sub_doc) and len(list(re.finditer("示例", sub_doc))) >= 5:
            message = "[Multi Task]Context中存在多个示例需要手工处理 index:%d \nsub_doc:%s" % (i, sub_doc)
            return message
        return None

    def deal_with_html(self, html_code, configs):
        have_repeat = 0
        ban_sub_doc_message = [[], []]
        flited_header, flited_context = fliter(html_code)
        flited_context_with_h1 = [flited_header+"\n"] + flited_context
        splited_doc = spliter(flited_context_with_h1, configs["rank_label"])
        QA_dict = {}
        Q_text_cnt = 0
        theme = self.fliter_html_label_all(flited_header).strip()
        if "：" in theme:
            theme = theme.split("：")[1]
        logger.info(f"[INFO] sub doc num: {len(splited_doc)}")
        for i, sub_doc in enumerate(splited_doc):
            check_message = self.check_sub_doc(i, sub_doc)
            if check_message:
                if "[Bad Context]" in check_message:
                    ban_sub_doc_message[0].append(check_message)
                elif "[Multi Task]" in check_message:
                    ban_sub_doc_message[1].append(check_message)
                continue
            sub_QA_dict = self.genertor.generateQA(sub_doc)
            hn_search = None
            for h_i in range(1, int(configs['rank_label'][1]) + 1, 1):
                search = re.search("<h{}>((?:.|\n)+)</h{}>".format(h_i, h_i), sub_doc)
                if not search:
                    break
                hn_search = search
            hn = ""
            if hn_search:
                hn = self.fliter_html_label_all(hn_search.group(1)).strip()
            sub_Q_text_cnt = 0
            for Q in sub_QA_dict.keys():
                if not self.check_question(Q) or not self.check_answer(sub_QA_dict[Q]):
                    continue
                if self.deal_Q(Q, theme, hn, sub_QA_dict[Q], QA_dict):
                    sub_Q_text_cnt += 1
            logger.info("[INFO] sub doc QA num: %d" % sub_Q_text_cnt)
            Q_text_cnt += sub_Q_text_cnt
        logger.info("[INFO] total QA num: %d" % Q_text_cnt)
        return QA_dict, have_repeat, ban_sub_doc_message

    def deal_with_url(self, url, configs):
        html_code = self.loader(url)
        QA_dict, have_repeat, additonal_message = self.deal_with_html(html_code, configs)
        _, title_l = re.search("<h1[^>]*>", html_code).span()
        _, title_r = re.search("</h1>", html_code).span()
        title = html_code[title_l:title_r-5]
        return QA_dict, have_repeat, additonal_message, title
    
    def run(self, html_dirs):
        result = {}
        for dir in html_dirs:
            try:
                with open(dir, 'r') as f:
                    html = f.read()
                QA_dict, have_repeat, additonal_message = self.deal_with_html(html, self.config)
                for q, a in QA_dict.items():
                    if q not in result or len(result[q])<a:
                        result[q] = a
            except Exception as e:
                logger.error(e)
        return result

    def del_model_cache(self):
        if self.config['LLM'] == 'Local':
            logger.info("Removing local llm cache from gpu memory.")
            self.genertor.llm.del_model_cache()
            logger.info("Clear finished.")

if __name__ == "__main__":
    x = HTML2QA()
    x.run()