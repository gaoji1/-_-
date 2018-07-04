# 包导入
import datetime
import json
import jpype
import math
import numpy as np
from sklearn.cluster import KMeans  # 导入K-means算法包


# 初始化JAVA虚拟机函数
def init_split_JVM():
    """初始化JVM，加载分词包，返回jvm"""
    print("医学分词包", ":", './wstool2.jar')
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % ('./wstool2.jar'))
    # sys.exit()
    jd = jpype.JClass("com.ifly.ti.ws2.WSEngineMgr")
    jd.init(u'ws.cfg', u'UTF-8')
    return jd


def inner_mean_distance(corpus_vector, kmeans):
    """计算类内平均距离，输入为俩个变量：1.文段的向量表示 2.kmeans的聚类结果类变量
       输出为一个列表result,其中result[i]表示第i类的类内平均距离"""
    # 聚类结果
    cluster_list = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    result = [0 for i in range(CLUSTER_NUM)]
    for i, item in enumerate(corpus_vector):
        # 当前corpus属于哪个类
        temp_class = cluster_list[i]
        result[temp_class] += np.sqrt(np.sum(np.square(item - cluster_centers[temp_class])))
    for i, item in enumerate(result):
        result[i] = item / np.sum(cluster_list == i)
    print("每个类的内部平均间距", ":", result)
    print("每个类包含的corpus个数", ":", [np.sum(cluster_list == i) for i in range(CLUSTER_NUM)])
    return result


def inner_distance_variance(corpus_vector, kmeans):
    """计算每一类的点到类中心的距离的方差,输入1.corpus向量，2.kmeans结果向量
       返回result为每一类的距离的方差"""
    cluster_list = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    means = [0 for i in range(CLUSTER_NUM)]  # 均值
    mean_square = [0 for i in range(CLUSTER_NUM)]  # 均方值
    result = [0 for i in range(CLUSTER_NUM)]  # 结果
    for i, item in enumerate(corpus_vector):
        # 当前corpus属于哪个类
        temp_class = cluster_list[i]
        means[temp_class] += np.sqrt(np.sum(np.square(item - cluster_centers[temp_class])))
        mean_square[temp_class] += np.sum(np.square(item - cluster_centers[temp_class]))
    for i in range(len(result)):
        means[i] = means[i] / np.sum(cluster_list == i)
        mean_square[i] = mean_square[i] / np.sum(cluster_list == i)
        result[i] = mean_square[i] - pow(means[i], 2)
    #     print("每个类的内部距离均方值",":",mean_square)
    print("每个类的内部距离方差", ":", result)
    print("每个类包含的corpus个数", ":", [np.sum(cluster_list == i) for i in range(CLUSTER_NUM)])
    return result


def label_distance(kmeans):
    """计算平均类间距离,输入为kmeans结果类,输出为平均类间距离result"""
    distance_list = []
    for i in range(len(kmeans.cluster_centers_)):
        for j in range(len(kmeans.cluster_centers_)):
            if j <= i:
                continue
            distance_list.append(np.sqrt(np.sum(np.square(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j]))))
    print("两类之间的距离最大为", ":", np.max(distance_list))
    print("两类之间的距离最小为", ":", np.min(distance_list))
    print("两类之间的距离平均为", ":", np.mean(distance_list))
    return np.mean(distance_list)


# 具体逻辑
def corpus_encoding(tf_idf_dic, corpus):
    """语段编码，输入两个参数分别是tf_idf字典跟语段corpus"""
    code = []
    total_sum = 0
    for item in tf_idf_dic:
        code.append(tf_idf_dic[item] * corpus.count(item))
        total_sum += tf_idf_dic[item] * corpus.count(item)
    if total_sum == 0:
        return code
    for i, item in enumerate(code):
        code[i] = item / total_sum
    return code


def add_dictionary(dic, item):
    """向字典中添加内容，如果内容已经在字典中，就+1，如果内同不在字典中，就初始化为1"""
    if item in dic:
        dic[item] += 1
    else:
        dic[item] = 1


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


def corpus_to_vector(number_of_corpus):
    # 先初始化分词工具
    split_tool = init_split_JVM()
    # 打开文件，计算tf-idf值，最终的目的是得到tf-idf值的字典
    with open("huxi_result_break.json", "r", encoding="utf8") as load_f:
        # 词数量，统计每个词在语料中出现的次数
        word_count = {}
        # 总共的词数量
        total_count = 0
        # 每个词的词频:tf
        word_frequency = {}
        # 总的语段个数
        total_corpus = 0
        # 有目标词的语段，统计当前词在几个语段中出现
        corpus_with_word = {}
        # idf字典
        idf_dic = {}
        # tf-idf字典
        tf_idf_dic = {}
        for i in range(number_of_corpus):
            line = load_f.readline()
            if not line:
                break
            if not isJson(line):
                continue
            # json转字符串
            json_to_dic = json.loads(line)
            # 总语段数＋1
            total_corpus += 1
            # 对现病史分词,去标点符号 ,去除长度为1的分词项
            split_result = split_tool.process(json_to_dic["xianbingshi"], 4)
            split_result = [item for item in split_result if not is_punctuation(item) and len(item) > 1]
            # 在当前语段中统计词出现个数
            for item in split_result:
                total_count += 1
                add_dictionary(word_count, item)
            # 在当前语段中统计词是否出现
            for item in set(split_result):
                add_dictionary(corpus_with_word, item)
        # 计算词频tf
        for item in word_count:
            word_frequency[item] = word_count[item]/total_count
        # 计算idf值
        for item in corpus_with_word:
            #只出现一次的就不要算了
            if corpus_with_word[item] <= 2:
                continue
            idf_dic[item] = math.log(total_corpus / (1 + corpus_with_word[item]))
        # 计算tf-idf
        for item in idf_dic:
            tf_idf_dic[item] = word_frequency[item] * idf_dic[item]
        # tf-idf取值最大的前1000项
        idf_dic = dict(sorted(idf_dic.items(), key=lambda x: x[1], reverse=True)[:1000])
        # tf_idf_dic = dict(sorted(tf_idf_dic.items(),key = lambda x:x[1],reverse = True)[:1000])
        load_f.close()
    print("词频列表",":",sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
    print("idf字典",":",idf_dic)
    # 打开文件，对于每一个corpus，计算它的编码
    with open("huxi_result_break.json", "r", encoding="utf8") as load_f:
        # 文本的编码
        corpus_code = []
        for i in range(number_of_corpus):
            line = load_f.readline()
            if not line:
                break
            if not isJson(line):
                continue
            # json转字符串
            json_to_dic = json.loads(line)
            corpus_code.append(corpus_encoding(idf_dic, json_to_dic["xianbingshi"]))
        print("number of corpus", ":", len(corpus_code))
        load_f.close()

    # 计算文本相似度矩阵
    similar_matrix = np.zeros((len(corpus_code), len(corpus_code)))
    print("begin filling the similar matrix", ":", datetime.datetime.now())
    for i in range(len(corpus_code)):
        for j in range(len(corpus_code)):
            similar_matrix[i][j] = np.sum(np.array(corpus_code[i]) * np.array(corpus_code[j]))
    print("end filling the similar matrix", ":", datetime.datetime.now())
    # 相似度矩阵的每一行作为相应corpus的向量vector
    print("shape of the similar matrix", ":", similar_matrix.shape)
    return similar_matrix
    jpype.shutdownJVM()

def Silhouette(X, y):
    """计算轮廓系数，输入1.corpus向量矩阵，2.corpus的label"""
    from sklearn.metrics import silhouette_samples, silhouette_score

    print('计算轮廓系数:')

    silhouette_avg = silhouette_score(X, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X, y)  # 每个点的轮廓系数

    return silhouette_avg, sample_silhouette_values

if __name__=="__main__":
    # 常量定义
    DATA_NUM = 2000  # 需要使用的数据数
    CLUSTER_NUM = 1000  # 聚类的数量

    # 文段转向量
    corpus_vector = corpus_to_vector(DATA_NUM)

    # 实行kmeans算法
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0).fit(corpus_vector)

    np.set_printoptions(threshold=np.inf)  # 全部输出
    # 把文件中的相应句子根据聚类结果输出
    with open("huxi_result_break.json", "r", encoding="utf8") as load_f:
        """把句子分门别类存放"""
        corpus_group = [[] for i in range(CLUSTER_NUM)]
        for i in range(len(corpus_vector)):
            line = load_f.readline()
            if not line:
                break
            #       每类所有句子输出
            corpus_group[kmeans.labels_[i]].append(line)
    #       每类出一个句子
    #         if len(corpus_group[kmeans.labels_[i]]) == 0:
    #             corpus_group[kmeans.labels_[i]].append(line)
    with open("temp.json", "w", encoding="utf8") as write_f:
        """写入文件temp.json"""
        for i, items in enumerate(corpus_group):
            write_f.write("=" * 100 + "\n")
            write_f.write("class" + str(i) + "\n")
            for j, item in enumerate(corpus_group[i]):
                json_to_dic = json.loads(item)
                write_f.write(json_to_dic["xianbingshi"] + "\n")
                write_f.write("\n")

    inner_mean_distance(corpus_vector, kmeans)
    inner_distance_variance(corpus_vector, kmeans)
    label_distance(kmeans)
    silhouette_avg, sample_silhouette_values = Silhouette(corpus_vector, kmeans.labels_)  # 平均轮廓系数，每个corpus的轮廓系数
    print("average Silhouette", ":", silhouette_avg)