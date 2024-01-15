import re

# 功能：获取当前行的最大优先等级，并设置优先级阈值，小于优先级的返回最低优先级-1
# 输入：
#    line:str 需要识别优先级的行
#    rank_threshold：str 优先级阈值对应的数值
# 返回：
#    max_rank:int 当前行的优先级数值
def get_line_rank(line, rank_key,rank_threshold):
    max_rank = 0
    for key in rank_key.keys():
        if "<%s" % key in line:
            max_rank = max(rank_key[key],max_rank)
    return max_rank if max_rank >= rank_threshold else 0

# 功能：计算文本中除了html标签和空格之外的字符长度
# 通过设置max_len，令代码只能过滤短于max_len的标签，防止误删
# 输入：
#   text:str 需要检查长度的字符串
#   max_len:int 过滤标签的最长长度，默认10
# 返回：
#   length:int 当前文本的长度
def get_text_length(text,max_len = 10):
    length = len(text)
    pattern = "<.{1,%d}>" % max_len
    html_label_pattern = re.compile(pattern)
    find_iter = list(html_label_pattern.finditer(text))
    for f_it in find_iter:
        l_p, r_p = f_it.span()[0], f_it.span()[1]
        length -= r_p - l_p if r_p - l_p <= max_len else 0
    return length

# 功能：通常情况下html过长，chatGPT无法正常理解，采用优先级（rank）单调栈的逻辑分割html源码
#   将第一个<h2>标签前的文本内容认为是本文摘要，赋予最高优先级
#   在没有拿到下一个相同或更高优先级的句子之前不断收集line，在拿到下一个相同或更高优先级的句子后，将这段时间收集的内容入栈
#   在遍历尾部增加向栈中加入优先级为6且内容为""的context，以保证在函数结束时栈中所有内容都已经出栈
# 输入：
#    doc_read_lines:list[str] list中每个元素代表html源码的一行
#    title:str html中的<header>……<header>部分
#    rank_label:str 切分html的最小切分单位，默认h2
# 返回：
#    sub_docs:list[str] 每个元素代表文本切分的一个部分，title+collection_context
def collecte_rank(doc_lines, rank_label="h2"):
    def get_text_collection(line_i):
        '''
            作用：在遇到下一个同级或更高级元素前，收集lines
            输入：当前line_id
            返回：收集到的lines，下一个同级元素的line_id
        '''
        context_collection = doc_lines[line_i]
        line_i += 1
        while line_i < len(doc_lines):
            line = doc_lines[line_i]
            line_rank = get_line_rank(line, rank_key, rank_key[rank_label])
            if line_rank != 0:
                break
            context_collection += line
            line_i += 1
        return context_collection, line_i

    rank_key = {"h1":5,"h2":4,"h3":3,"h4":2,"h5":1,"#":0}
    stack_rank = [float("inf")]
    stack_text = [""]

    sub_contexts = []
    line_i = 0
    while line_i < len(doc_lines):
        line = doc_lines[line_i]
        cur_rank = get_line_rank(line, rank_key, rank_key[rank_label])
        context_collection, line_i = get_text_collection(line_i)
        if not stack_rank or cur_rank < stack_rank[-1]: # 仍未切换元素，仍在递归进入更细粒度的元素
            stack_rank.append(cur_rank)
            stack_text.append(context_collection)
        else:
            pop_context = []    # 存储同级元素和所有父级元素
            while stack_rank and cur_rank >= stack_rank[-1]:    # 弹出同级元素
                stack_rank.pop(-1)
                pop_context.append(stack_text.pop(-1))
            for i in range(len(stack_text)-1,-1,-1):
                pop_context.append(stack_text[i])
            sub_context = "".join(pop_context[::-1])    # 上一个同级元素及其父级元素即为新的chunk
            if len(sub_context) > 0:
                sub_contexts.append(sub_context)
            stack_rank.append(cur_rank) # 新的元素入栈
            stack_text.append(context_collection)
    if stack_rank:
        sub_contexts.append("".join(stack_text))    # 最后一个chunk
    return sub_contexts

def spliter(filted_context,rank_label):
    return collecte_rank(filted_context,rank_label)