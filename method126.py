from sklearn.externals import joblib
import random
import numpy
from math import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot  as plt
#按比例提取 ：round
def bili(num_list,tq_num,R):
    #print(tq_num,num_list)
    sum_num=sum(num_list)
    data_Q=num_list
    r = R
    Qi = []
    Qisum = 0
    ks = 0
    while Qisum != tq_num:
        if ks == 0:
            Qi = []
            for lda_Q in data_Q:
                if (round(tq_num * lda_Q/ sum_num, 0)) >= 2:
                    Qi.append(int(round(tq_num * lda_Q / sum_num, 0)))
                elif round(tq_num * lda_Q / sum_num, 3) >= (tq_num * R / (len(data_Q))):
                    Qi.append(1)
                else:
                    Qi.append(0)
        Qisum = 0
        #print(Qi)
        for i in Qi:
            Qisum += i
        Qimax_index = Qi.index(max(Qi))
        Qimin_index = int(random.random()*len(data_Q))
        #print(Qisum, ',', R, ',', tq_num, ',', Qi[Qimax_index])
        if Qisum > tq_num and Qi[Qimax_index] > (Qisum - tq_num):
            Qi[Qimax_index] = int(Qi[Qimax_index] + tq_num - Qisum)
            break
        elif Qisum < tq_num and data_Q[Qimin_index]>Qi[Qimin_index]:
            # Qi[Qimax_index] = int(Qi[Qimax_index] + tq_num - Qisum)
            Qi[Qimin_index] = int(Qi[Qimin_index] + 1)
        elif Qi[Qimax_index] <= (Qisum - tq_num):
            R += random.random() * r
            Qi[Qimax_index] = int(Qi[Qimax_index] - 1)
        if R > (Qisum - tq_num):
            ks = ks + 1
    return Qi


def sim(P, Q):
    sumpq = 0
    sump2 = sum([P[i] ** 2 for i in range(len(P))])
    sumq2 = sum([Q[i] ** 2 for i in range(len(Q))])
    for i in range(len(P)):
        sumpq += P[i] * Q[i]
    up = sumpq
    down = (sump2 * sumq2) ** 0.5
    try:
        sim_data = up / down
    except:
        sim_data = 1
    return sim_data


def m_cov(data, data_list):
    sim_data = 0
    for data_x in data_list:
        sim_data += sim(data, data_x)
    return sim_data


def b_reundance(center_data, a_cluster_data, q_cluster_data, geshu, num_list, Weight):
    cluster_x_data = []
    a_cluster_r_data = []
    q_cluster_r_data = []
    cluster_x_data.append(num_list[center_data])
    num_list.remove(num_list[center_data])
    del (a_cluster_data[center_data])
    del (q_cluster_data[center_data])
    Rk = len(cluster_x_data)
    while Rk < int(geshu):
        max_redundance_data = 0
        max_redundance_k = 0
        sum_redundance = 0
        for data_k in range(len(a_cluster_data)):
            a_data_i = a_cluster_data[data_k]
            q_data_i = q_cluster_data[data_k]
            sum_sim_ij = 0
            for data_j in range(len(a_cluster_r_data)):
                a_data_j = a_cluster_r_data[data_j]
                q_data_j = q_cluster_r_data[data_j]
                sum_sim_ij += (1 - Weight) * m_cov(a_data_i, a_data_j) + Weight * m_cov(q_data_i, q_data_j)
                try:
                    sum_redundance += (1 - 1.0 / sum_sim_ij)
                except:
                    sum_redundance += 0
            if max_redundance_data < (sum_redundance * 1.0 / Rk):
                max_redundance_data = sum_redundance * 1.0 / Rk
                max_redundance_k = data_k
        cluster_x_data.append(num_list[max_redundance_k])
        num_list.remove(num_list[max_redundance_k])
        del (a_cluster_data[max_redundance_k])
        del (q_cluster_data[max_redundance_k])
        Rk = len(cluster_x_data)
    return cluster_x_data


def tq_method(hang_all_list, a_data_list, q_data_list, tq_num_list, Weight):
    tq_result_list = []
    for num in range(len(tq_num_list)):
        K = tq_num_list[num]
        num_list = hang_all_list[num]
        if K > 1:
            a_data = a_data_list[num]
            q_data = q_data_list[num]
            # 最大代表性的点
            max_r_cov_num = 0
            max_num = 0
            for r_num in range(len(a_data)):
                similarities_number = (1 - Weight) * m_cov(q_data[r_num], q_data) + Weight * m_cov(a_data[r_num],
                                                                                                   a_data)
                if similarities_number > max_num:
                    max_num = similarities_number
                    max_r_cov_num = r_num


            # 冗余度提取
            tq_result_list += b_reundance(max_r_cov_num, list(a_data), list(q_data), K, list(num_list), Weight)
        elif K == 1:
            tq_result_list.append(num_list[0])
    return tq_result_list

#数据整理阶段-----------------
#最底层神经源
questions_growing_indexs=[2,4,7,8]
answers_growing_indexs=[2,4,5,6]
som_model_dir="./som_model/"
som=joblib.load(som_model_dir+"som.model")
q_data=som.questions_input_vects
a_data=som.answers_input_vects
qa_indexs=[]
num_qa_list=[]
all_qa_index=[]
for i in range(som.m):
    for j in range(som.n):
        if len(som.qa[i,j])>4:
            qa_indexs.append((i,j))
        num_qa_list.append(len(som.qa[i,j]))
        all_qa_index.append((i,j))
qa_indexs=[(2,2),(4,2),(4,4),(7,6),(8,4),(8,5)]
hang_num_list=[]
hang_all_list=[]
hang_q_vect_list=[]
hang_a_vect_list=[]
for i in range(som.m):
    #qa_all_list=[]
    for j in range(som.n):
        qa_som_list = []
        hang_som_list=[]
        a_vect_list=[]
        q_vect_list=[]
        num_list=[]
        if (i,j) in qa_indexs and i in questions_growing_indexs and j in answers_growing_indexs:
            qa_som_name="qa_cluster_growing"+str(i)+","+str(j)+".model"
            qa_som=joblib.load(som_model_dir+qa_som_name)
            #qa_som_set[i,j]=qa_som
            #print(i,j,qa_som.m,qa_som.n,qa_som.qa)

            for x in range(qa_som.m):
                for y in range(qa_som.n):
                    if len(qa_som.qa[x,y])>0:
                        #print(qa_som.qa)
                        #print(x,y,qa_som.qa[x,y])
                        #print(numpy.array(som.qa[i,j])[qa_som.qa[x,y]])
                        qa_som_list+=list(numpy.array(som.qa[i,j])[qa_som.qa[x,y]])
                        hang_som_list.append(list(numpy.array(som.qa[i,j])[qa_som.qa[x,y]]))
        else:
            if len(som.qa[i,j])>0:
                #print(som.qa[i,j])
                qa_som_list+=som.qa[i,j]
                hang_som_list.append(som.qa[i,j])
        for som_x in hang_som_list:
            a_vect_list.append(som.answers_input_vects[som_x])
            q_vect_list.append(som.questions_input_vects[som_x])
            num_list.append(len(som_x))

        #qa_all_list.append(qa_som_list)
        hang_num_list.append(num_list)
        hang_all_list.append(hang_som_list)
        hang_a_vect_list.append(a_vect_list)
        hang_q_vect_list.append(q_vect_list)
def listFlatten(src):
    tmp = []
    for i in src:
        if type(i) is not list:
            tmp.append(i)
        else:
            tmp.extend(listFlatten(i))
    return tmp


def cal_cover(re_result, data):
    """
    计算覆盖度
    :param re_result: 代表性问答对的集合
    :param data: 原始的问答对编号集合
    :return: 代表性问答对相对于原始集合的覆盖度
    """
    sum_cov = 0
    for data_index in data:
        di=data_index
        max_sim = np.float("-inf")
        for re in re_result:
            dj = re
            sim_dij =  sim(di, dj)
            if sim_dij > max_sim:
                max_sim = sim_dij
        sum_cov += max_sim
    # print("提取代表性问答对的覆盖度为：", sum_cov / len(data))
    # print(len(data))
    return sum_cov / len(data)

def cal_reund(re_result):
    """
    计算冗余度
    :param re_present: 代表性问答对
    :return: 冗余度
    """
    sum_red=0.0
    for i in range(len(re_result)):
        sum_sim=0.0
        di= re_result[i]
        for j in range(len(re_result)):
            dj= re_result[j]
            sum_sim+=sim(di,dj)
        sum_red+=(1-1/sum_sim)
    #print("提取代表性问答对的冗余度为：",sum_red/len(re_result))
    return sum_red/len(re_result)
    # q_re_sim = np.sum(cosine_similarity(re_present_q), axis=0)
    # # np.sum(1-1/q_re_sim)
    # a_re_sim = np.sum(cosine_similarity(re_present_a), axis=0)
    # # np.sum(1-1/a_re_sim)
    # sum_red = (1 - Weight) * np.sum(1 - 1 / q_re_sim) + Weight * np.sum(1 - 1 / a_re_sim)
    # print("提取的代表性问答对的冗余度为：", sum_red / len(re_result))
    # reundance.append(sum_red / len(re_result))
    # return sum_red / len(re_result)

def cal_coverage(re_result, data):
    """
    计算覆盖度
    :param re_result: 代表性问答对的集合
    :param data: 原始的问答对编号集合
    :return: 代表性问答对相对于原始集合的覆盖度
    """
    sum_cov = 0
    for data_index in data:
        di_q = q_data[data_index]
        di_a = a_data[data_index]
        max_sim = np.float("-inf")
        for re in re_result:
            dj_q = q_data[re]
            dj_a = a_data[re]
            sim_dij = (1 - Weight) *sim(di_a, dj_a)  + Weight * sim(di_q, dj_q)
            if sim_dij > max_sim:
                max_sim = sim_dij
        sum_cov += max_sim
    print("提取代表性问答对的覆盖度为：", sum_cov / len(data))
    print(len(data))
    return sum_cov / len(data)
def cal_reundance(re_result):
    """
    计算冗余度
    :param re_present: 代表性问答对
    :return: 冗余度
    """
    re_present_q = q_data[re_result]
    re_present_a = a_data[re_result]
    sum_red=0.0
    for i in range(len(re_present_q)):
        sum_sim=0.0
        sum_sim_q=0.0
        sum_sim_a=0.0
        di_q = re_present_q[i]
        di_a = re_present_a[i]
        for j in range(len(re_present_q)):
            dj_q = re_present_q[j]
            dj_a = re_present_a[j]
            sum_sim_q+=sim(di_q,dj_q)
            sum_sim_a+=sim(di_a,dj_a)
        sum_sim+=(1-Weight)*sum_sim_a+Weight*sum_sim_q
        sum_red+=(1-1/sum_sim)
    print("提取代表性问答对的冗余度为：",sum_red/len(re_result))
    return sum_red/len(re_result)
    # q_re_sim = np.sum(cosine_similarity(re_present_q), axis=0)
    # # np.sum(1-1/q_re_sim)
    # a_re_sim = np.sum(cosine_similarity(re_present_a), axis=0)
    # # np.sum(1-1/a_re_sim)
    # sum_red = (1 - Weight) * np.sum(1 - 1 / q_re_sim) + Weight * np.sum(1 - 1 / a_re_sim)
    # print("提取的代表性问答对的冗余度为：", sum_red / len(re_result))
    # reundance.append(sum_red / len(re_result))
    # return sum_red / len(re_result)
#main开始输入点[30,50,70]#999
rates=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
coverage=[]
reundance=[]
# global q_data
# global a_data
# q_data=som.questions_input_vects
# a_data=som.answers_input_vects
for rate in rates:
    print("提取的比例为：",rate)
    tq_all_num=round(331*rate)
    R=0.1
    Weight=0.5
    re_result=[]
    tq_num_list=bili(num_qa_list,tq_all_num,R)
    #print(num_qa_list)
    print(hang_all_list)
    print("提取开始---")
    for x_num in range(len(hang_all_list)):
        #print(hang_num_list[x_num],hang_all_list[x_num],len(hang_a_vect_list[x_num]),len(hang_q_vect_list[x_num]))
        tq_bili_list=bili(hang_num_list[x_num],tq_num_list[x_num],R)

        if sum(tq_bili_list)>0:
            tq_result_list=tq_method(hang_all_list[x_num],hang_a_vect_list[x_num],hang_q_vect_list[x_num],tq_bili_list,Weight)
            #print()
            print("第",str(x_num),"神经元需提取个数为：",str(sum(tq_bili_list)),"代表为：", tq_result_list)
            data_source=listFlatten(hang_all_list[x_num])
            cov=cal_coverage(tq_result_list,data_source)
            reu=cal_reundance(tq_result_list)
            q_cover=cal_cover(q_data[tq_result_list],q_data[data_source])
            a_cover=cal_cover(a_data[tq_result_list],a_data[data_source])
            q_redu=cal_reund(q_data[tq_result_list])
            a_redu=cal_reund(a_data[tq_result_list])
            print("问题的覆盖度为：",q_cover,"\t","问题的冗余度为：",q_redu)
            print("答案的覆盖度：",a_cover,"\t","答案的冗余度为：",a_redu)
            re_result+=tq_result_list
        else:
            print("第", str(x_num), "神经元需提取个数为：", str(sum(tq_bili_list)))

    print("提取结束----最终提取的代表结果如下：------")
    print(re_result)
    print(len(re_result))
    print(len(set(re_result)))
    print("计算提取的代表性问答对的覆盖度....")
    sum_cov=0
    q_data=som.questions_input_vects
    a_data=som.answers_input_vects
    re_present_q=q_data[re_result]
    re_present_a=a_data[re_result]

# list_q_data=list(q_data)
# list_a_data=list(a_data)
# for re_index in re_result:
#     re_q_data=q_data[re_index]
#     re_a_data=a_data[re_index]
#     list_q_data.remove(re_q_data)
#     list_a_data.remove(re_a_data)
#     list_q_data.append(re_q_data)
#     list_a_data.append(re_a_data)
#     q_max_sim=np.max(cosine_similarity(list_q_data)[:,-1][:-1])
#     a_max_sim=np.max(cosine_similarity(list_a_data)[:,-1][:-1])





    for data_index in range(len(q_data)):
        di_q=q_data[data_index]
        di_a=a_data[data_index]
        max_sim=np.float("-inf")
        for re in re_result:
            dj_q=q_data[re]
            dj_a=a_data[re]
            sim_dij=(1-Weight)*sim(di_a,dj_a)+Weight*sim(di_q,dj_q)
            if sim_dij>max_sim:
                max_sim=sim_dij
        sum_cov+= max_sim
    print("提取代表性问答对的覆盖度为：",sum_cov/len(q_data))
    coverage.append(sum_cov/len(q_data))
    print("计算提取的代表行问答对的冗余度....")
    re_present_q=q_data[re_result]
    re_present_a=a_data[re_result]
    q_re_sim=np.sum(cosine_similarity(re_present_q),axis=0)
    #np.sum(1-1/q_re_sim)
    a_re_sim=np.sum(cosine_similarity(re_present_a),axis=0)
    #np.sum(1-1/a_re_sim)
    avg_red=Weight*np.sum((1-1/q_re_sim)/len(re_result))+(1-Weight)*np.sum((1-1/a_re_sim)/len(re_result))
    print("提取的代表性问答对的冗余度为：",avg_red)
    reundance.append(avg_red)

    avg_red=cal_reundance(re_result)
    print("cal_reundance计算的冗余度为：",avg_red)
    q_cover = cal_cover(q_data[re_result], q_data)
    a_cover = cal_cover(a_data[re_result], a_data)
    q_redu = cal_reund(q_data[re_result])
    a_redu = cal_reund(a_data[re_result])
    print("问题的覆盖度为：", q_cover, "\t", "问题的冗余度为：", q_redu)
    print("答案的覆盖度：", a_cover, "\t", "答案的冗余度为：", a_redu)

print("覆盖度：",coverage)
print("冗余度：",reundance)
plt.plot(rates,coverage,color="red",label="coverage",linewidth=1.0)
plt.plot(rates,reundance,color="blue",label="reudance",linewidth=1.0)
plt.show()







