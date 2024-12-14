import pyhanlp
import re
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
import json


month = sys.argv[1]

# demo_date = "0517"
# demo_date = "0529"
demo_date = sys.argv[2]

words_dict = {}
with open("Setences_pattern/words4sent.txt") as F:
    for line in F:
        w = line.strip().split("\t")
        words_dict[w[0]] = int(w[1])
words_list = np.array([x[0] for x in sorted(words_dict.items(),key=lambda x: x[1])])

MAX_LEN = 4
fast_sents = []
fast_tmp = {}
with open("Setences_pattern/fast_sentences.txt") as F:
    for line in F:
        sent = np.zeros(MAX_LEN)
        line = line.strip().split("\t")
        sent[:len(line)-3] = [int(float(x)) for x in line[3:]] 
        fast_tmp[len(fast_sents)] = line[:3]
        fast_sents.append(sent)
fast_sents = np.array(fast_sents).astype(int)

slow_sents = []
slow_tmp = {}
with open("Setences_pattern/slow_sentences.txt") as F:
    for line in F:
        sent = np.zeros(MAX_LEN)
        line = line.strip().split("\t")
        sent[:len(line)-3] = [int(float(x)) for x in line[3:]] 
        slow_tmp[len(slow_sents)] = line[:3]
        slow_sents.append(sent)
slow_sents = np.array(slow_sents).astype(int)

puncs = re.compile(r"\s|\.|\(|\)|"+"|".join([":",",","，","。","、","：","；","？","！","（","）","《","》","#",
                         "-","——","·","……","/",r"\\","\\[","\\]","【","】","\\|","℃","—>","\n",
                                            "\?","!","…","@"]))

exclude_list = ['快乐', '快递', '快感', '快门', '快餐', '快活', '快手', '快鱼', '快子', '痛痛快快',
                '大快人心', '乘龙快婿', '遂心快意', '称心快意', '口快心直', '轻轻快快', '轻车快马', '愉快', '痛快',
                '欢快', '爽快', '轻快', '明快', '畅快', '勤快', '凉快', '捕快', '外快', '一快', # 快
                    "小时候", # 小时
                    '天津', '天空', '天使', '天堂', '天气', '天然', '天生', '天才', '天涯', '天真', '天王',
                    '天线', '天河', '天赋', '天后', '天价', '天籁', '天性', '天大', '天子', '天边', '天文',
                    '天际', '天亮', '天外', '天意', '天国', '天道', '天府', '天平', '天鹅', '天桥', '天仙',
                    '天皇', '天书', '天窗', '天池', '天尊', '天命', '新天地', '老天爷', '全天候', '破天荒',
                    '满天飞', '打天下', '小天地', '航天器', '南天门', '倚天剑', '拜天地', '摩天楼', '信天翁',
                    '信天游', '顶天立地','聊天', '上天', '航天', '老天', '漫天', '飞天', '九天', '先天', '晴天', '冲天',
                    '露天', '惊天', '朝天', '普天', '通天', '苍天', '云天', '升天', '青天', '逆天', '乐天',
                    '西天', '参天', '滔天', '开天', '回天', '南天', '翻天', # 天
                    '月亮', '月饼', '月球', '月影', '月色', '月夜', '月下', '月季','月华', '月牙', '月老', '月宫',
                    '月桂', '月坛', '月轮', '月相', '月晕', '月兔', '风月','新月', '冷月', '水月', '望月', '皓月', 
                    '残月', '赏月', '圆月', '奔月', '偃月', '皎月','盈月', # 月
                    '年轻', '年龄', '年纪', '年级', '年会', '年少', '年青', '年幼', '年迈', '年长', '年货',
                    '年糕', '年画', '年轮', '年假', '更年期', '万年历', '少年宫', '编年史',
                    '跨年度', '大年夜', '忘年交', '贺年卡', '上年纪', '青年', '少年', '中年', '童年', '过年',
                    '老年', '成年', '晚年', '拜年', '幼年', '光年', '壮年', '英年', # 年
                    '周围', '周边', '周期', '周到', '周密', '周遭', '周旋', '周身', '周易', '周转', '周公',
                    '周全', '周瑜', '周折', '周游', '周长', '周济', '周知', '周正', '圆周率', '牙周炎', '不周',
                    '西周', '圆周', '商周', '东周', '北周', '后周', '抓周', # 周
                    "傲慢","怠慢","时间别太长"] # https://cidian.qianp.com/zuci/ 中挑选与时间无关的常见词组

file = os.path.join("../../../Weibo-COV_V2_Dataset/Weibo-COV_V2_clean",month+".csv")
fast_weibo = []
slow_weibo = []
hopes = ["想要","希望","真想","拜托","如果","假如"]
os.makedirs("../output/%s_demo/Weibo-COV_V2_fast"%(demo_date),exist_ok=True)
os.makedirs("../output/%s_demo/Weibo-COV_V2_slow"%(demo_date),exist_ok=True)
F_fast = open(os.path.join("../output/%s_demo/Weibo-COV_V2_fast"%(demo_date),month+".csv"),"w")
F_slow = open(os.path.join("../output/%s_demo/Weibo-COV_V2_slow"%(demo_date),month+".csv"),"w")
cnt_f,cnt_s,cnt_all = 0,0,0
with open(file) as F:
    title = F.readline()
    F_fast.write("template_id,template,key,"+title)
    F_slow.write("template_id,template,key,"+title)
    title = title.strip().split(",")
    title_idx = dict(zip(title,np.arange(len(title))))
    for line_o in tqdm(F):
        try:
            cnt_all += 1
            line = line_o.strip().split(",")
            content = line[title_idx['content']]
            content = content.replace("\"","").replace("\'","").replace(",","，")
            if '转发理由:' in content:
                content = content.split("转发理由:")[1]
            if not len(content):
                continue
            in_f, in_s = False, False
            f_pos, s_pos = 0,0
            sentences = [s for s in puncs.split(content) if len(s)]
            for s in sentences:
                w_in = np.array([True]+[w in s for w in words_list])
                if not max(w_in):
                    continue
                for w in exclude_list:                                 
                    s = s.replace(w,"~")
                if not in_f:
                    f_all = w_in[fast_sents].min(axis=1)
                    if f_all.max():
                        no_sort = True
                        for idx in np.where(f_all==True)[0]:
                            index = -1
                            for w in list(fast_sents[idx])+[0]:
                                if w==0:
                                    if fast_tmp[idx][0] == '2.0': # 特殊处理
                                        hope_indexs = [s.index(w) for w in hopes if w in s]
                                        if len(hope_indexs):
                                            print(s)
                                        if len(hope_indexs) and min(hope_indexs) < s.index(words_list[fast_sents[idx][0]]):
                                            print(s)
                                            continue
                                        else:
                                            in_f = True
                                            break
                                    else:
                                        in_f = True
                                        break
                                n_index = s.index(words_list[w-1])
                                if n_index >index:
                                    index = n_index
                                else:
                                    break
                            if in_f:
                                f_pos = idx
                                break
                if not in_s:
                    s_all = w_in[slow_sents].min(axis=1)
                    if s_all.max():
                        no_sort = True
                        for idx in np.where(s_all==True)[0]:
                            index = -1
                            for w in list(slow_sents[idx])+[0]:
                                if w==0:
                                    if slow_tmp[idx][0] == "2.0": # 特殊处理
                                        hope_indexs = [s.index(w) for w in hopes if w in s]
                                        if len(hope_indexs):
                                            print(s)
                                        if len(hope_indexs) and min(hope_indexs) < s.index(words_list[slow_sents[idx][0]]):
                                            continue
                                        else:
                                            in_s = True
                                            break
                                    else:
                                        in_s = True
                                        break
                                n_index = s.index(words_list[w-1])
                                if n_index >index:
                                    index = n_index
                                else:
                                    break
                            if in_s:
                                s_pos = idx
                                break
            if in_f:
                tmp = fast_tmp[f_pos]
                # ws = " ".join(words_list[[p for p in fast_sents[f_pos]-1 if p>=0]])
                line[title_idx['content']] = content
                F_fast.write(",".join(tmp)+","+",".join(line)+"\n")
                cnt_f += 1
                if cnt_f % 200 == 0:
                    print("CNT Fast: %d / %d (%.2f%%)"%(cnt_f,cnt_all,cnt_f/cnt_all*100))
            if in_s:
                tmp = slow_tmp[s_pos]
                # ws = " ".join(words_list[[p for p in slow_sents[s_pos]-1 if p>=0]])
                line[title_idx['content']] = content
                F_slow.write(",".join(tmp)+","+",".join(line)+"\n")
                cnt_s += 1

                if cnt_s % 20 == 0:
                    print("CNT Slow: %d / %d (%.2f%%)"%(cnt_s,cnt_all,cnt_s/cnt_all*100))
            # if cnt_s>=20:
            #     break
            
        except:
            continue

F_fast.close()
F_slow.close()


print("CNT All: %d, slow: %d (%.2f%%), fast: %d (%.2f%%) "%(cnt_all,cnt_s,cnt_s/cnt_all*100,cnt_f,cnt_f/cnt_all*100))

