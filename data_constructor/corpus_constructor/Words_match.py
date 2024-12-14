import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
import json
from harvesttext import HarvestText
from tqdm import tqdm
from harvesttext.resources import get_qh_typed_words
import sys
import re

# hyper params
version = sys.argv[1]
month = sys.argv[2]

# Sentence split tools
ht = HarvestText()
typed_words = get_qh_typed_words()
ht.add_typed_words(typed_words)
ht.add_new_words(['时光飞逝', '时间飞逝', '白驹过隙', '一眨眼', '光阴似箭', '岁月如梭', '时光匆匆', '时间匆匆', '如梭', '转眼', '日月如梭', '转瞬即逝', '一下子', '一晃', '如驹', '过隙', '即逝', '一转眼', '时光荏苒', '一晃而过', '忽忽', '光阴似箭', '光阴如箭', '光阴', '度日如年', '漫长', '漫漫', '漫漫长夜', '这么久', '三秋', '日长似岁', '漫长', '度日如年', '一日三秋', '寸阴若岁', '时间过得慢', '过得慢'])
puncs = re.compile(r"\s|\.|\(|\)|"+"|".join([":",",","，","。","、","：","；","？","！","（","）","《","》","#",
                         "-","——","·","……","‘","’","“","”","/",r"\\","\\[","\\]","【","】","\\|","℃","—>","\n"]))
def split_weibo(weibo, do=0):
    all_sentences = []
    sentences = ht.cut_sentences(weibo)
    for sen in sentences:
        sen_split = []
        for param in puncs.split(sen):
            if len(param):
                sen_split+=ht.seg(param,stopwords=["全文"],return_sent=False)
        if len(sen_split):
            if do:
                print(sen_split)
            all_sentences+=sen_split
    return all_sentences

# load words to match
# words_f_file = "模型扩展词汇表/扩展词典-第二次_hpx/words_fast_%s.txt"%(version)
words_f_file = "Words_pattern/words_fast_0606.txt"
words_fast = []
with open(words_f_file) as f:
    for line in f:
        words_fast.append(line.strip())

# words_s_file = "模型扩展词汇表/扩展词典-第二次_hpx/words_slow_%s.txt"%(version)
words_s_file = "Words_pattern/words_slow_0606.txt"
words_slow = []
with open(words_s_file) as f:
    for line in f:
        words_slow.append(line.strip())

words_fast_set = set(words_fast)
words_slow_set = set(words_slow)

# Load all weibo and match
file = os.path.join("../../Weibo-COV_V2_Dataset/Weibo-COV_V2_clean",month+".csv")
os.makedirs("output/%s/Weibo-COV_V2_slow"%(version),exist_ok=True)
os.makedirs("output/%s/Weibo-COV_V2_fast"%(version),exist_ok=True)
F_out_slow = open(os.path.join("output/%s/Weibo-COV_V2_slow"%(version),month+".csv"),"w")
F_out_fast = open(os.path.join("output/%s/Weibo-COV_V2_fast"%(version),month+".csv"),"w")
F = open(file)

head = F.readline().strip("\n").split(",")
head_dict = dict(zip(head,range(len(head))))
F_out_slow.write(",".join(["key"]+head)+"\n")
F_out_fast.write(",".join(["key"]+head)+"\n")

s_cnt = 0
f_cnt = 0
all_cnt = 0
for line in tqdm(F):
    raw_line = line.strip("\n").split(",")
    if len(raw_line) == 11: # 特殊处理，如果origin和geo_info都有
        raw_line[-2] = raw_line[-2]+"，"+raw_line[-1]
        raw_line = raw_line[:-1]
    try:
        raw_weibo = raw_line[head_dict['content']]
    except:
        continue
    raw_weibo = raw_weibo.replace("\"","").replace("\'","").replace(",","，")
    if '转发理由:' in raw_weibo:
        raw_weibo = raw_weibo.split("转发理由:")[1]
        raw_line[head_dict['content']] = raw_weibo
    all_cnt += 1
    f_w = []
    s_w = []
    for w in words_fast:
        if w in raw_weibo:
            f_w.append(w)
    for w in words_slow:
        if w in raw_weibo:
            s_w.append(w)
    if len(f_w):
        s = set(split_weibo(raw_weibo))
        words =  s & words_fast_set
        if len(words):
            F_out_fast.write(",".join(["&".join(list(words))]+raw_line)+"\n")
            f_cnt += 1
    if len(s_w):
        s = set(split_weibo(raw_weibo))
        words = s&words_slow_set
        if len(words):
            F_out_slow.write(",".join(["&".join(list(words))]+raw_line)+"\n")
            s_cnt += 1
            
F.close()
F_out_slow.close()
F_out_fast.close()

print("All: %d, Fast: %d (%.3f%%), Slow: %d (%.3f%%)"%(all_cnt,f_cnt,f_cnt/all_cnt*100,s_cnt,s_cnt/all_cnt*100))