# -*- coding: utf-8 -*-
"""包导入"""
from simhash import Simhash
import jieba
import jieba.posseg as pseg
from jieba import analyse
import numpy as np
import os
import matplotlib.pyplot as plt 
import matplotlib as mpl
import json
from math import log
import re
import datetime
import logging


"""使用simhash的方法计算相似度"""
def isJson(jsonstr):
    """判断是否是json"""
    try:
        a = json.loads(jsonstr)
        return True
    except:
        return False

logging.basicConfig(level=logging.NOTSET)  # 设置日志级别    
with open("all_result_break.json",'r',encoding="utf8") as load_f:  
    xianbingshi = []
    for i in range(150000):
        #准备工作
        line = load_f.readline()
        if not line:
            break
        if not isJson(line):
            continue;
        json_to_dic = json.loads(line)
        
        #存储现病史字符串
        if "subject_name" not in json_to_dic:
            continue
        if "呼吸" not in json_to_dic["subject_name"]:
            continue
        if "xianbingshi" not in json_to_dic:
            continue
        temp_xianbingshi = json_to_dic["xianbingshi"]
        xianbingshi.append(temp_xianbingshi)
    
    #计算相似度,打印相似度较大的句子
    simhash_list = [] # 里面存储着每一个句子对应的simhash对象
    print("现病史数据个数",":",len(xianbingshi))
    print(datetime.datetime.now())
    for index,item in enumerate(xianbingshi):
        temp_simhash_obj = Simhash(item)
        simhash_list.append(temp_simhash_obj)
    print(datetime.datetime.now())
    for i_x,x in enumerate(simhash_list):
        print("#"*50)
        print("现在寻找与此句相似的xianbingshi",":",xianbingshi[i_x])
        for i_y,y in enumerate(simhash_list):
            if i_y<=i_x:
                continue
            if(x.distance(y)<25):
                print(xianbingshi[i_y])
                print(x.distance(y))
                print("-"*50)
    print(datetime.datetime.now())
                
    load_f.close()