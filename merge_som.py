from sklearn.externals import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from SOM_cos import SOM

def merge_questions(som):
    """
    合并som问题维度神经元
    :param som: 需要合并的som网络
    :return: 合并完成后的som网络
    """
    tau_2 = 0.05
    questions=list(som.questions_input_vects)
    questions.append(np.mean(som.questions_input_vects,axis=0))
    dis=cosine_distances(questions)
    questions_mqe0=np.sum(dis[:,-1])
    print("tau_2*questions_mqe0:",tau_2*questions_mqe0)
    print(som.questions_mqe)
    print(som.questions_mqe>tau_2*questions_mqe0)
    index = 0
    while index<som.m:
        print("index:",index)
        if index+1<som.m:
            # print(som.questions_dataitems[index] + som.questions_dataitems[index + 1])
            # print(np.mean(som.questions_weights[index:index + 2], axis=0))
            # merge_mqe = np.sum(cosine_distances(
            #     list(som.questions_dataitems[index] + som.questions_dataitems[index + 1]).append(
            #         np.mean(som.questions_weights[index:index + 2], axis=0)))[:, -1])
            if som.questions_mqe[index]+som.questions_mqe[index+1]<tau_2*questions_mqe0:
                print("需要合并的两神经元：",index,index+1)
                if som.questions_mqe[index+1]==0:
                    som.questions_weights[index]=som.questions_weights[index]
                    som.questions_mqe[index]=som.questions_mqe[index]
                elif som.questions_mqe[index]==0:
                    som.questions_weights[index] = som.questions_weights[index+1]
                    som.questions_mqe[index]=som.questions_mqe[index+1]
                else:
                    som.questions_weights[index]=np.mean(som.questions_weights[index:index+2],axis=0)
                    som.questions_mqe[index]=np.mean([som.questions_mqe[index],som.questions_mqe[index+1]])
                som.questions_dataitems[index].extend(som.questions_dataitems[index+1])
                som.questions_dataitems[index + 1]=[]
                som.questions_labels[index].extend(som.questions_labels[index+1])
                som.questions_labels[index + 1]=[]
                som.questions_weights=np.delete(som.questions_weights,index+1,axis=0)
                som.questions_mqe=np.delete(som.questions_mqe,index+1,axis=0)
                som.m=len(som.questions_weights)
                # som.calculate_mqe()
            else:
                index+=1
        else:
            index += 1
        #som.projector_questions(som.questions_input_vects)
    new_som=SOM.SOM(som.questions_input_vects,som.answers_input_vects,som.m,som.n,1000, 0.3,  "gaussian",
                     som.layer, None, None)
    new_som.answers_weights=som.answers_weights
    new_som.questions_weights=som.questions_weights
    #new_som.questions_train(som.questions_input_vects)

    new_som.project_data(som.questions_input_vects,som.answers_input_vects)
    new_som.calculate_mqe()
    return new_som

def merge_answers(som):
    """
    合并som的答案维度
    :param som: 需要合并的som网络
    :return: 合并完成的som网络
    """
    tau_2 = 0.05
    answers = list(som.answers_input_vects)
    print(type(som.answers_input_vects))
    print(np.array(som.answers_input_vects).shape)
    answers.append(np.mean(som.answers_input_vects, axis=0))
    dis = cosine_distances(answers)
    answers_mqe0 = np.sum(dis[:, -1])
    print("tau_2 * answers_mqe0:",tau_2 * answers_mqe0)
    print(som.answers_mqe)
    print(som.answers_mqe>tau_2 * answers_mqe0)
    index = 0
    while index < som.n:
        print("index:",index)
        if index + 1 < som.n:
            # merge_mqe=np.sum(cosine_distances(list(som.answers_dataitems[index]+som.answers_dataitems[index+1]).append(np.mean(som.answers_weights[index:index + 2], axis=0)))[:,-1])
            if som.answers_mqe[index]+som.answers_mqe[index+1] < tau_2 * answers_mqe0:
                print("需要合并的两神经元：", index, index + 1)
                if som.answers_mqe[index+1]==0:
                    som.answers_weights[index]=som.answers_weights[index]
                    som.answers_mqe[index]=som.answers_mqe[index]
                elif som.answers_mqe[index]==0:
                    som.answers_weights[index] = som.answers_weights[index+1]
                    som.answers_mqe[index]=som.answers_mqe[index+1]
                else:
                    som.answers_weights[index] = np.mean(som.answers_weights[index:index + 2], axis=0)
                    som.answers_mqe[index]=np.mean([som.answers_mqe[index],som.answers_mqe[index+1]])
                som.answers_dataitems[index].extend(som.answers_dataitems[index + 1])
                som.answers_dataitems[index + 1]=[]
                som.answers_labels[index].extend(som.answers_labels[index + 1])
                som.answers_labels[index + 1]=[]
                som.answers_weights = np.delete(som.answers_weights, index + 1, axis=0)
                som.answers_mqe=np.delete(som.answers_mqe, index + 1, axis=0)
                som.n= len(som.answers_weights)
                # som.calculate_mqe()

            else:
                index += 1
        else:
            index += 1
    #som.projector_answers(som.answers_input_vects)
    new_som = SOM.SOM(som.questions_input_vects, som.answers_input_vects, som.m, som.n, 1000, 0.3, "gaussian",
                      som.layer, None, None)
    new_som.answers_weights = som.answers_weights
    new_som.questions_weights = som.questions_weights
    #new_som.answers_train(som.answers_input_vects)
    new_som.project_data(som.questions_input_vects, som.answers_input_vects)
    new_som.calculate_mqe()
    return new_som

qa_som_list={}#由qa位置生长的网络的集合，其中key为上层神经元的的坐标，value为由该坐标生成的网络
q_som_list={}#由questions位置生长的网络集合，其中key为上一层神经元的坐标，value为由该坐标生产的网络
a_som_list={}#由answers位置生长的网络集合，其中key为上一层神经元的坐标，value为由该坐标生产的网络
questions_growing_indexs=[0,2,4,5,7]#第一层网络问题坐标需要生长的神经元编号
answers_growing_indexs=[0,1,2,5,6,7,9,10]#第一层为昂罗答案坐标需要生长的答案神经元编号
som_model_dir="../data_output/net/"#som持久化的文件路径
merge_model_dir="../data_output/net/merge/"
som=joblib.load(som_model_dir+"som.model")#加载第一层网络，som为第一层网络


tau_2 = 0.05
questions=list(som.questions_input_vects)
questions.append(np.mean(som.questions_input_vects,axis=0))
dis=cosine_distances(questions)
questions_mqe0=np.sum(dis[:,-1])
print("tau_2*questions_mqe0:",tau_2*questions_mqe0)
print(som.questions_mqe)
print(som.questions_mqe>tau_2*questions_mqe0)

answers = list(som.answers_input_vects)
print(type(som.answers_input_vects))
print(np.array(som.answers_input_vects).shape)
answers.append(np.mean(som.answers_input_vects, axis=0))
dis = cosine_distances(answers)
answers_mqe0 = np.sum(dis[:, -1])

print("tau_2*answers_mqe0:",tau_2*answers_mqe0)
print(som.answers_mqe)
print(som.answers_mqe>tau_2*answers_mqe0)

# som.m#为som问题维度神经元个数
# som.n#为som答案维度神经元个数
# som.questions_input_vects#som的问题维度数据
# som.answers_input_vects#som的答案维度数据
# som.questions_labels#类型是键值对，key为问题维度神经元编号，value为该神经元映射的问题编号
# som.answers_labels#类型是键值对，key为答案维度神经元编号，value为该神经元映射的答案编号
# som.questions_dataitems#类型是键值对，key为问题维度神经元编号，value为映射到该神经元的问题对应的tfidf
# som.answers_dataitems#类型是键值对，key为答案维度神经元编号，value为映射到该神经元的答案对应的tfidf
# som.questions_dataitems_count#类型是键值对，key为答案维度神经元编号，value为映射到该神经元的问题数量
# som.answers_dataitems_count#类型是键值对，key为答案维度神经元编号，value为映射到该神经元的答案数量
# som.qa#类型为键值对，key为问答对的位置，value为映射到该位置的问答对编号
# som.qa_count#类型为键值对，key为问答对的位置，value为映射到该位置的问答对数量
# som.questions_mqe##类型是键值对，key为问题维度神经元编号，value为该神经元问题的量化误差
# som.answers_mqe##类型是键值对，key为答案维度神经元编号，value为该神经元答案的量化误差
# som.layer#som在第一层
qa_indexs=[]#存储第一层网络中qa位置神经元上映射数据集大于4的qa位置编号
for i in questions_growing_indexs:
    q_som_name="questions_cluster_growing"+str(i)+".model"
    q_som=joblib.load(som_model_dir+q_som_name)
    q_som_list[i]=q_som
    print("问题维度网络"+str(i)+"在合并之前的结构为：",q_som.m,",",q_som.n)
    print("合并之前的questions_labels:",q_som.questions_labels)
    print("合并之前的answers_labels:", q_som.answers_labels)
    merge_som=merge_answers(q_som)
    print("问题维度网络" + str(i) + "在合并之后的结构为：", merge_som.m, ",", merge_som.n)
    print("合并之后的questions_labels:", merge_som.questions_labels)
    print("合并之后的answers_labels:", merge_som.answers_labels)
    joblib.dump(merge_som,merge_model_dir+"questions"+str(i)+".model",compress=3)

for j in answers_growing_indexs:
    a_som_name="answers_cluster_growing"+str(j)+".model"
    a_som=joblib.load(som_model_dir+a_som_name)
    a_som_list[j]=a_som
    print("答案维度网络" + str(j) + "在合并之前的结构为：", a_som.m, ",", a_som.n)
    print("合并之前的questions_labels:", a_som.questions_labels)
    print("合并之前的answers_labels:", a_som.answers_labels)
    merge_som = merge_questions(a_som)
    print("问题维度网络" + str(j) + "在合并之后的结构为：", merge_som.m, ",", merge_som.n)
    print("合并之后的questions_labels:", merge_som.questions_labels)
    print("合并之后的answers_labels:", merge_som.answers_labels)
    joblib.dump(merge_som, merge_model_dir + "answers" + str(j) + ".model", compress=3)

for i in range(som.m):
    for j in range(som.n):
        if len(som.qa[i,j])>4:
            qa_indexs.append((i,j))
for i in questions_growing_indexs:
    for j in answers_growing_indexs:
        if (i,j) in qa_indexs:
            qa_som_name="qa_cluster_growing"+str(i)+","+str(j)+".model"
            qa_som=joblib.load(som_model_dir+qa_som_name)
            qa_som_list[i,j]=qa_som
            print("qa网络" +str(i)+","+ str(j) + "在合并之前的结构为：", qa_som.m, ",", qa_som.n)
            print("合并之前的questions_labels:", qa_som.questions_labels)
            print("合并之前的answers_labels:", qa_som.answers_labels)
            merge_som = merge_questions(qa_som)
            merge_som = merge_answers(merge_som)
            print("qa网络" + str(i) + "," + str(j) + "在合并之后的结构为：", merge_som.m, ",", merge_som.n)
            print("合并之后的questions_labels:", merge_som.questions_labels)
            print("合并之后的answers_labels:", merge_som.answers_labels)
            joblib.dump(merge_som, merge_model_dir + "qa" + str(i) +","+str(j)+ ".model", compress=3)


