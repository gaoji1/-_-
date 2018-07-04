#coding=utf-8
#包导入
import datetime
import json
import jpype
import math
import numpy as np
from sklearn.cluster import KMeans  # 导入K-means算法包
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import Birch

# 初始化JAVA虚拟机函数
def init_split_JVM():
    """初始化JVM，加载分词包，返回jvm"""
    print("医学分词包", ":", './wstool2.jar')
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % ('./wstool2.jar'))
    # sys.exit()
    jd = jpype.JClass("com.ifly.ti.ws2.WSEngineMgr")
    jd.init(u'ws.cfg', u'UTF-8')
    return jd

def is_punctuation(split_item):
    """检测分词后的任意一项是不是一个标点符号"""
    split_item = split_item.replace(" ", "")
    if split_item in ",.，。、:：“ ”\"\"()（） ":
        return True
    else:
        return False

def isJson(jsonstr):
    """判断是否是json"""
    try:
        a = json.loads(jsonstr)
        return True
    except:
        return False

def read_n_line(file_name,num_of_line):
    """读取某个文件的前n行，输入为两个值，1.file_name文件名 2.num_of_line读取的行数
       返回为resultt，result每一项为一行"""
    with open(file_name,"r",encoding="utf8") as file:
        result = []
        for i in range(num_of_line):
            single_line = file.readline()
            if not single_line:
                break
            result.append(single_line)
        file.close()
    return result

def split_word(corpus,split_tool=init_split_JVM()):
        """分词函数，输入为1.分词工具，这里其实就是讯飞的医学分词工具，2.语段，就是一个字符串
            返回result，也是一个字符串，但是这个字符串中把分出来的词之间用空格分隔开"""
        stop_words = open("stop_words.txt","r",encoding="utf8").readlines()
        for  i,item in enumerate(stop_words):
            stop_words[i] = stop_words[i].replace("\n","")
        split_result = split_tool.process(corpus,4) #分词结果，这是一个列表
        split_result = [item for item in split_result if not is_punctuation(item) and len(item) > 1 and item not in stop_words] #去除标点符号
        result = " ".join(split_result)
        return result

def split_word_batch(corpus_list):
    """批量对corpus进行分词，输入1.corpus列表，每项是一个corpus，
        返回result为一个列表，每一项为一个分词后的corpus string"""
    result = []
    for corpus in corpus_list:
        single_result = split_word(corpus)
        result.append(single_result)
    return result

def json_2_string(patient_jsonstring):
    """读入的是病例数据的jsonstring，把它转化成字典，提取出里边的'现病史'字符串"""
    json_to_dic = json.loads(patient_jsonstring)
    return json_to_dic["xianbingshi"]

def TFIDF(corpus_token_list):
    """计算tf-idf值，输入1.corpus_token_list，分词后的corpus序列
        输出：由tfidf计算出的corpus表示矩阵"""
    print("begin calculate tf-idf")
    # 将文本中的词语转换成词频矩阵,矩阵元素 a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()

    # 该类会统计每个词语tfidf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_token_list))

    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    return weight

def PCA(weight,dimension):
    """PCA降维，输入1.weight为corpus向量矩阵，2.dimentson为需要讲到多少维度"""
    from sklearn.decomposition import PCA

    print('原有维度: ', len(weight[0]))
    print('开始降维:')

    pca = PCA(n_components=dimension)  # 初始化PCA
    X = pca.fit_transform(weight)  # 返回降维后的数据
    print('降维后维度: ', len(X[0]))

    return X


def Silhouette(X, y):
    """计算轮廓系数，输入1.corpus向量矩阵，2.corpus的label"""
    from sklearn.metrics import silhouette_samples, silhouette_score

    print('计算轮廓系数:')

    silhouette_avg = silhouette_score(X, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X, y)  # 每个点的轮廓系数

    return silhouette_avg, sample_silhouette_values

if __name__=="__main__":
    DATA_NUM = 2000
    CLUSTER_NUM = 900
    patient_info = read_n_line("huxi_result_break.json",DATA_NUM) #读数据
    corpus_list = [json_2_string(patient_jsonstring) for patient_jsonstring in patient_info] #提取现病史数据
    corpus_token_list = split_word_batch(corpus_list)   #分词
    weight = TFIDF(corpus_token_list)   #计算corpus向量，利用tfidf
    #weight = PCA(weight,dimension=1000) #进行pcd降维
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0).fit(weight)  #使用kmeans聚类
    brich = Birch(n_clusters=CLUSTER_NUM).fit(weight) #使用birch聚类
    silhouette_avg, sample_silhouette_values = Silhouette(weight, kmeans.labels_)  # 平均轮廓系数，每个corpus的轮廓系数
    print("="*30,"kmeans","="*30)
    print("average Silhouette",":",silhouette_avg)
    # print("Silhouette for each",":",sample_silhouette_values)
    print("="*30,"birch","="*30)
    silhouette_avg, sample_silhouette_values = Silhouette(weight, brich.labels_)  # 平均轮廓系数，每个corpus的轮廓系数
    print("average Silhouette", ":", silhouette_avg)
    # print("Silhouette for each", ":", sample_silhouette_values)

    with open("huxi_result_break.json", "r", encoding="utf8") as load_f:
        """把句子分门别类存放"""
        corpus_group = [[] for i in range(CLUSTER_NUM)]
        for i in range(len(weight)):
            line = load_f.readline()
            if not line:
                break
            #       每类所有句子输出
            corpus_group[kmeans.labels_[i]].append(line)
    #       每类出一个句子
    #         if len(corpus_group[kmeans.labels_[i]]) == 0:
    #             corpus_group[kmeans.labels_[i]].append(line)
    with open("temp_02.txt", "w", encoding="utf8") as write_f:
        """写入文件temp.json"""
        for i, items in enumerate(corpus_group):
            write_f.write("=" * 100 + "\n")
            write_f.write("class" + str(i) + "\n")
            for j, item in enumerate(corpus_group[i]):
                json_to_dic = json.loads(item)
                write_f.write(json_to_dic["xianbingshi"] + "\n")
                write_f.write("\n")

    jpype.shutdownJVM()