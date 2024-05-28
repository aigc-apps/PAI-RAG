import re


# 功能：使用re.search正则搜索text中是否存在aim，仅限于搜索必须存在的项，若不存在aim则报错（RuntimeError）
# 传入：
#   aim:str 搜索目标 在调用时用来搜索类似<h2>的常见html标签
#   text:str 搜索文本 在该字符串中搜索目标
# 返回：
#   rs.span():tuple 搜索结果 aim出现的index，即text[rs.span()[0],rs.span()[1]] == aim
def re_search_raise_error(aim, text):
    rs = re.search(aim, text)
    if rs is None:
        raise RuntimeError("[NO Aim Error]html中没有目标内容 aim:{}".format(aim))
    else:
        return rs.span()


# 功能：使用re.search正则搜索text中是否存在aim，若不存在aim则返回None 若存在则返回搜索对象的左指针（l_p）和右指针（r_p）
# 传入：
#   aim:str 搜索目标 在调用时用来搜索类似<h2>的常见html标签
#   text:str 搜索文本 在该字符串中搜索目标
# 返回：
#   不存在时：None
#   存在时：(l_p, r_p)
def re_search(aim, text):
    rs = re.search(aim, text)
    if rs is None:
        return rs
    else:
        return rs.span()


# 针对<header>中的<div class="BreadCrumb--breadcrumb--prPHyhx">……</div>部分包含的内容执行过滤
def check_html_code_header(BreadCrumb):
    # 检查是否包含以下关键词，有的话就报错
    ban_keywords = (
        "常见问题",
        "产品简介",
        "发布记录",
        "开通地域",
        "发行版本",
        "发展历程",
        "客户案例",
        "相关协议",
        "概述",
        "公告",
        "视频",
        "附录",
    )
    for bkw in ban_keywords:
        if isinstance(bkw, str):
            if bkw in BreadCrumb:
                raise RuntimeError(
                    "[Bad Header]html的header中存在需要过滤的内容 ban:%s" % str(bkw)
                )
                # return False, bkw
        else:
            # 所有的子关键字都出现时才return False
            flag = True
            for sub_kw in bkw:
                flag &= sub_kw in BreadCrumb
            if flag:
                raise RuntimeError(
                    "[Bad Header]html的header中存在需要过滤的内容 ban:%s" % str(bkw)
                )
                # return not flag, bkw
    return True, None


# 功能：
#  html中普遍使用<header>……</header>标记header部分
# 使用<div .*class=\"markdown-body\"标记正文部分
# 传入：
#    html_code:str 字符串格式的html源码
# 返回：
#    header:str <header> …… </header>部分
#    context:str <div .*class=\"markdown-body\" …… </div>部分
def cut_context(html_code):
    # 默认结构：<header> breadcrumb <h1> title </h1> </header>
    # 找 header 中的 h1
    header_start, _ = re_search_raise_error("<header", html_code)
    html_code = html_code[header_start:]
    h1_start, _ = re_search_raise_error("<h1", html_code)
    # breadcrumb = html_code[:h1_start]
    html_code = html_code[h1_start:]
    _, h1_end = re_search_raise_error("</h1>", html_code)
    title = html_code[:h1_end]
    _, header_end = re_search_raise_error("</header>", html_code)
    # check_html_code_header(breadcrumb+title)

    # 找 context
    html_code = html_code[header_end:]
    context_start, _ = re_search_raise_error('<div .*class="markdown-body"', html_code)
    # 因为部分 <div> 被跳过了，所以后面的部分 </div> 会比 <div> 多
    # 下面要做的就是通过贪心找到 <div markdown-body> 对应的 </div> 位置
    html_code = html_code[context_start:]
    div_list = list(re.finditer("<div", html_code))
    re_div_list = list(re.finditer("</div>", html_code))
    label_stack = [1]
    i, j = 0, 0
    while label_stack:
        if div_list[i].span()[0] < re_div_list[j].span()[0]:
            label_stack.append(1)
            i += 1
        else:
            label_stack.pop(-1)
            j += 1
    _, context_end = re_div_list[j - 1].span()
    context = html_code[:context_end]
    re_search_raise_error("<h2", context)  # 检查是否包含h2元素
    return title, context


# 功能：过滤html标签中的属性部分
#   针对<img>标签 仅保留src所表示的图片链接部分
#   将包含<h2>标签的行在行位增加换行符号"\n"
#   将html代码中的空格替换词全部换成空格
#   去掉只由空格" "，制表符"\t"，换行符"\n"组成的行
# 传入：
#    html_context:str 切分后的html源码
# 返回：
#    filtered_html_lines:list[str] 按行分开的过滤后的html源码（分割不严谨）
def filter_html_code(
    html_code,
    space_blocks=("&rsquo;", "&#39;", "&#34;", "&nbsp;", "&amp;", "&lt;", "&gt;"),
):
    def filter_param(text):
        pattern = re.compile("<[^/>]* [^>]+>")
        f_it = list(pattern.finditer(text))
        for f_i in range(len(f_it) - 1, -1, -1):
            l_p, r_p = f_it[f_i].span()[0], f_it[f_i].span()[1]
            while text[l_p] != " ":
                l_p += 1
            if "img" == text[f_it[f_i].span()[0] + 1 : l_p]:
                continue
            if "a" == text[f_it[f_i].span()[0] + 1 : l_p]:
                continue
            text = text[:l_p] + text[r_p - 1 :]
        return text

    def filter_image(text):
        img_pattern = re.compile('<img [^>]*src="')
        f_it = list(img_pattern.finditer(text))
        for f_i in range(len(f_it) - 1, -1, -1):
            l_p, r_p = f_it[f_i].span()
            link_l_p = link_r_p = r_p
            while text[link_r_p] != '"':
                link_r_p += 1
            img_link = text[link_l_p:link_r_p]
            r_p = link_r_p
            while text[r_p] != ">":
                r_p += 1
            text = "{} {} {}".format(text[:l_p], img_link, text[r_p + 1 :])
        return text

    def filter_a(text):
        img_pattern = re.compile('<a [^>]*href="')
        f_it = list(img_pattern.finditer(text))
        for f_i in range(len(f_it) - 1, -1, -1):
            l_p, r_p = f_it[f_i].span()
            link_l_p = link_r_p = r_p
            while text[link_r_p] != '"':
                link_r_p += 1
            href_link = text[link_l_p:link_r_p]
            r_p = link_r_p
            while text[r_p] != ">":
                r_p += 1
            text_r_p = r_p + 1
            while text_r_p < len(text) and text[text_r_p] != "<":
                text_r_p += 1
            text = "{}{} https://help.aliyun.com{} {}".format(
                text[:l_p], text[r_p + 1 : text_r_p], href_link, text[text_r_p:]
            )
        return text

    html_lines = html_code.split("\n")
    filtered_html_lines = []
    for line in html_lines:
        line = filter_param(line)  # 清除标签属性和参数
        line = filter_image(line)  # 清除img标签
        line = filter_a(line)  # 清除超链接
        # 替换空格的替换词（替换HTML转义字符）
        for space_block in space_blocks:
            line = line.replace(space_block, " ")
        # 去掉只由空格" "，制表符"\t"，换行符"\n"组成的行
        if not re.search("[^\s]", line):
            continue
        # 在”代码如下“后加上”示例“（line.replace('代码如下', '代码如下示例')就行了）
        code_follow_items = [f_it.span() for f_it in re.finditer("代码如下", line)]
        for it_i in range(len(code_follow_items) - 1, -1, -1):
            line = (
                line[: code_follow_items[it_i][1]]
                + "示例"
                + line[code_follow_items[it_i][1] :]
            )
        # 部分html在读取后会无法正常换行，为了应对这种情况
        # 给每个<h2>标签前加上\n
        # 保证每个<h2>都在独立的一行
        hn_items = [f_it.span() for f_it in re.finditer("<h\d>", line)]
        l_point = 0
        if hn_items:
            filtered_html_lines.append(
                line[l_point : hn_items[0][0]] + "\n"
            )  # 第一个h标签之前的内容，拼上\n
            l_point = hn_items[0][0]
            hn_items_i = 1
            while hn_items_i < len(hn_items):
                hn_item = hn_items[hn_items_i]
                # 把 h2 “：”前的内容筛掉
                temp_line = line[l_point : hn_item[0]]
                h2_search = re.search(
                    r"<h\d>(?:.|\n)*?([^>]+)：([^<]+)(?:.|\n)*</h\d>", temp_line
                )
                if h2_search and len(h2_search.groups()) >= 2:
                    temp_line = "{}{}（{}）{}".format(
                        temp_line[: h2_search.span()[0] + 4],
                        h2_search.group(2).strip(),
                        h2_search.group(1).strip(),
                        temp_line[h2_search.span()[1] - 5 :],
                    )
                filtered_html_lines.append(temp_line + "\n")
                l_point = hn_item[0]
                hn_items_i += 1
        # 如果行内没有<h\d>且最后一个字符不是换行符就在其后加入一个换行符
        if not line.endswith("\n"):
            # 把 h2 “：”前的内容筛掉
            # 即 <h2>A：B</h2> 变为 <h2>B（A）</h2>
            temp_line = line[l_point:]
            h2_search = re.search(r"<h\d>.*?([^>]+)：([^<]+).*</h\d>", temp_line)
            if h2_search:
                temp_line = "{}{}（{}）{}".format(
                    temp_line[: h2_search.span()[0] + 4],
                    h2_search.group(2),
                    h2_search.group(1),
                    temp_line[h2_search.span()[1] - 5 :],
                )
            filtered_html_lines.append(temp_line + "\n")
    return filtered_html_lines


def filter_html(html_data):
    header, context = cut_context(html_data)
    filtered_header = "".join(filter_html_code(header))
    filtered_context = filter_html_code(context)
    return filtered_header, filtered_context
