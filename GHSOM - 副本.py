from SOM_cos import SOM
import pandas as pd
import numpy as np
from SOM_cos import LabelSOM
from SOM_cos import RepresentQAs
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn import *
class GHSOM(object):

    def __init__(self,questions_input_vects,answers_input_vects,tau_1,tau_2,neighborhood,alpha,q_word,a_word):
        """
        :param questions_input_vects: 问题维度输入数据
        :param answers_input_vects: 答案维度输入数据
        :param tau_1: 水平生长的参数
        :param tau_2: 垂直生长的参数
        :param N: 领域
        """
        self.questions_input_vects = questions_input_vects
        self.answers_input_vects = answers_input_vects
        if tau_1 is None:
            self.tau_1=0.3
        else:
            self.tau_1=float(tau_1)

        if tau_2 is None:
            self.tau_2=0.03
        else:
            self.tau_2=float(tau_2)
        if neighborhood is None:
            self.neighborhood="gaussian"
        else:
            self.neighborhood=neighborhood
        if alpha is None:
            self.alpha=0.3
        else:
            self.alpha=float(alpha)
        self.layer=1
        #计算第0层的平均量化误差
        self.questions_mqe0=self.calculate_questions_mqe0(self.questions_input_vects)
        self.answers_mqe0=self.calculate_answers_mqe0(self.answers_input_vects)
        self.qa_mqe0=0.5*self.questions_mqe0+0.5*self.answers_mqe0
        self.q_word=q_word
        self.a_word=a_word

    def calculate_questions_mqe0(self,questions_input_vects):
        """
        计算并放回问题维度的mqe0
        :param questions_input_vects: 输入问题数据
        :return: 0层问题维度的mqe
        """
        # dataframe = pd.DataFrame(questions_input_vects)
        # dataframe_mean = dataframe.mean()
        # dist = np.sum(np.power(dataframe - dataframe_mean, 2), axis=1)
        # question_mqe0 = np.mean(dist)
        sum=0
        for i in range(len(questions_input_vects)):
            sum+=1-self.get_distance(questions_input_vects[i],np.mean(questions_input_vects,axis=0))
        avg=np.mean(sum)
        question_mqe0=avg
        return question_mqe0

    def calculate_answers_mqe0(self,answers_input_vects):
        """
        计算并返回答案维度的mqe0
        :param answers_input_vects: 输入答案维度数据
        :return: 0层答案维度的mqe
        """
        # dataframe = pd.DataFrame(answers_input_vects)
        # dataframe_mean = dataframe.mean()
        # dist = np.sum(np.power(dataframe - dataframe_mean, 2), axis=1)
        # answer_mqe0 = np.mean(dist)
        sum = 0
        for j in range(len(answers_input_vects)):
            sum += 1-self.get_distance(answers_input_vects[j], np.mean(answers_input_vects, axis=0))
        avg = np.mean(sum)
        answer_mqe0 = avg
        return answer_mqe0

    def get_distance(self, input_vect, weight):
        """
        计算样本与神经元权重的距离
        :param input_vect: 样本
        :param weight: 权重
        :return: 距离
        """
        # 余弦相似度计算
        dist = np.sum(np.array(input_vect) * np.array(weight)) / (
        np.sqrt(np.sum(np.array(input_vect) ** 2)) * np.sqrt(np.sum(np.array(weight) ** 2)))
        return dist
    def questions_horizontal_growing(self,som):
        """
        问题维度的水平生长，此时答案维度保持不变
        :param som: 当前的需要进行问题维度生长的som
        :return:返回增加一行以后的som
        """
        max_question_mqe_index = som.get_max_question_mqe_index()
        som.m+=1
        som.questions_mqe = np.zeros(som.m)
        som.answers_mqe   =np.zeros(som.n)
        som.qa = {}
        som.questions_labels = {}
        som.answers_labels={}
        som.qa_count = np.zeros([som.m, som.n])
        for i in range(som.m):
            for j in range(som.n):
                som.qa[(i, j)] = []
        for i in range(som.m):
            som.questions_dataitems[(i)]=[]
            som.questions_labels[i]=[]
            som.questions_dataitems_count[(i)]=0
        for j in range(som.n):
            som.answers_dataitems[(j)] = []
            som.answers_dataitems_count[(j)] = 0
            som.answers_labels[j]=[]

        if max_question_mqe_index==som.m-1:
            som.questions_weights=np.insert(som.questions_weights,max_question_mqe_index,np.mean(som.questions_weights[max_question_mqe_index-1:max_question_mqe_index+1,:],axis=0),axis=0)
        else:
            som.questions_weights = np.insert(som.questions_weights, max_question_mqe_index+1, np.mean(som.questions_weights[max_question_mqe_index:max_question_mqe_index + 2, :], axis=0), axis=0)

        som.questions_train(som.questions_input_vects)
        som.project_data(som.questions_input_vects, som.answers_input_vects)
        som.calculate_mqe()

    def answers_horizontal_growing(self,som):
        """
        答案维度的水平生长，此时问题维度保持不变
        :param som: 需要进行答案维度生长的som
        :return: 增加一列以后的som
        """
        max_answer_mqe_index = som.get_max_answer_mqe_index()
        som.n+=1
        som.questions_mqe = np.zeros(som.m)
        som.answers_mqe   =np.zeros(som.n)
        som.qa = {}
        som.questions_labels = {}
        som.answers_labels = {}
        som.qa_count = np.zeros([som.m, som.n])
        for i in range(som.m):
            for j in range(som.n):
                som.qa[(i, j)] = []
        for j in range(som.n):
            som.answers_dataitems[(j)] = []
            som.answers_labels[j]=[]
            som.answers_dataitems_count[(j)] = 0
        for i in range(som.m):
            som.questions_dataitems[(i)] = []
            som.questions_labels[i] = []
            som.questions_dataitems_count[(i)] = 0

        if max_answer_mqe_index == som.n - 1:
            som.answers_weights = np.insert(som.answers_weights, max_answer_mqe_index, np.mean(
                som.answers_weights[max_answer_mqe_index - 1:max_answer_mqe_index + 1, :], axis=0), axis=0)
        else:
            som.answers_weights = np.insert(som.answers_weights, max_answer_mqe_index + 1, np.mean(
                som.answers_weights[max_answer_mqe_index:max_answer_mqe_index + 2, :], axis=0), axis=0)

        som.answers_train(som.answers_input_vects)
        som.project_data(som.questions_input_vects, som.answers_input_vects)
        som.calculate_mqe()

    def questions_vertical_growing(self,som,i):
        """
        问题垂直生长
        :param som: 当前的som
        :param i: 需要向下扩展的问题维度神经元编号
        :return: 下一层新生成的som
        """
        # 1.构建新的网络，其中答案维度保持不变
        # (self, input_vects, m, n, iterations, alpha, neighborhood, layer, row_index, col_index)
        next_answers_input_vects=som.answers_input_vects[som.questions_labels[i]]
        new_som = SOM.SOM(np.array(som.questions_dataitems[i]), np.array(next_answers_input_vects), 2, som.n, 1000, 0.3, self.neighborhood,
                         self.layer+1, None, None)
        new_som.answers_weights=som.answers_weights
        # 训练SOM网络
        new_som.questions_train(np.array(som.questions_dataitems[i]))
        new_som.project_data(new_som.questions_input_vects, new_som.answers_input_vects)
        new_som.calculate_mqe()
        # 计算当前网络的量化均方误差
        q_MQE = new_som.questions_MQE
        # 判断条件是否需要水平生长
        if q_MQE >= self.tau_1 * som.questions_mqe[i]:
            while q_MQE >= self.tau_1 * som.questions_mqe[i]:
                self.questions_horizontal_growing(new_som)
                q_MQE = new_som.questions_MQE
        #print("sublsyer labels.........")
        # x = {}
        # y = {}
        # for i in range(new_som.m):
        #     x[(i)] = LabelSOM.LabelSOM().label_som(new_som.questions_weights[i], new_som.questions_dataitems[(i)],
        #                                          self.q_word,
        #                                          0.005,
        #                                          0.05)
        # for j in range(new_som.n):
        #     y[(j)] = LabelSOM.LabelSOM().label_som(new_som.answers_weights[j], new_som.answers_dataitems[(j)],
        #                                          self.a_word, 0.005,
        #                                          0.05)
        #print(x)
        #print(y)
        #print("sublayer每一个聚类簇的结果。。。。。。。")
        #print("问题维度聚类簇。。。。。")
        #print(new_som.questions_labels)
        #print("答案维度聚类簇。。。。。")
        #print(new_som.answers_labels)
        return new_som

    def answers_vertical_growing(self,som,j):
        """
        答案维度垂直生长
        :param som: 当前的som
        :param j: 需要向下扩展的答案维度的神经元编号
        :return: 下一层新生成的som
        """
        # 1.构建新的网络，其中问题维度保持不变
        # (self, input_vects, m, n, iterations, alpha, neighborhood, layer, row_index, col_index)
        next_questions_input_vects=som.questions_input_vects[som.answers_labels[j]]
        new_som = SOM.SOM(np.array(next_questions_input_vects), np.array(som.answers_dataitems[j]), som.m, 2, 1000, 0.3,
                             self.neighborhood,
                             self.layer + 1, None, None)
        new_som.questions_weights = som.questions_weights
        # 训练SOM网络
        new_som.answers_train(np.array(som.answers_dataitems[j]))
        new_som.project_data(new_som.questions_input_vects, new_som.answers_input_vects)
        new_som.calculate_mqe()
        # 计算当前网络的量化均方误差
        a_MQE = new_som.answers_MQE
        # 判断条件是否需要水平生长
        if a_MQE >= self.tau_1 * som.answers_mqe[j]:
            while a_MQE >= self.tau_1 * som.answers_mqe[j]:
                self.answers_horizontal_growing(new_som)
                a_MQE = new_som.answers_MQE
        #print("sublsyer labels.........")
        # x = {}
        # y = {}
        # for i in range(new_som.m):
        #     x[(i)] = LabelSOM.LabelSOM().label_som(new_som.questions_weights[i], new_som.questions_dataitems[(i)], self.q_word,
        #                                          0.005,
        #                                          0.05)
        # for j in range(new_som.n):
        #     y[(j)] = LabelSOM.LabelSOM().label_som(new_som.answers_weights[j], new_som.answers_dataitems[(j)], self.a_word, 0.005,
        #                                          0.05)
        #print(x)
        #print(y)
        #print("sublayer每一个聚类簇的结果。。。。。。。")
        #print("问题维度聚类簇。。。。。")
        #print(new_som.questions_labels)
        #print("答案维度聚类簇。。。。。")
        #print(new_som.answers_labels)

        return new_som

    def qa_vertical_growing_som(self,som,i,j,questions_weights,answers_weights):
        """
        问答对生长
        :param som: 上一层网络
        :param i: 需要生长的问答对位置的横坐标
        :param j: 需要生长的问答对位置的纵坐标
        :param questions_weights: som问题维度神经元i垂直生长后的问题维度权重
        :param answers_weights: som答案维度神经元j垂直生长后的答案维度权重
        :return: qa对生长完成的som网络
        """
        questions_input_vects=som.questions_input_vects[som.qa[i,j]]
        answers_input_vects=som.answers_input_vects[som.qa[i,j]]
        m=len(questions_weights)
        n=len(answers_weights)
        iterations=1000
        alpha=0.3
        qa_som = SOM.SOM(np.array(questions_input_vects), np.array(answers_input_vects), m, n, iterations, alpha,
                      self.neighborhood, self.layer, None, None)
        #使新构建网络的问题维度神经元权重等于问题维度生长的问题维度神经元权重，
        #答案维度神经元等于答案维度生长的答案维度神经元权重
        qa_som.questions_weights=questions_weights
        qa_som.answers_weights=answers_weights
        qa_som.project_data(qa_som.questions_input_vects,qa_som.answers_input_vects)
        qa_som.calculate_mqe()
        return qa_som
    def is_need_questions_vertical_growing(self,som):
        for i in range(som.m):
            if som.questions_mqe[i] >= self.tau_2 * self.questions_mqe0:
                return True
        return False

    def is_need_answers_vertical_growing(self, som):
        for j in range(som.n):
            if som.answers_mqe[j] >= self.tau_2 * self.answers_mqe0:
                return  True
        return False

    def labelsAndVisalize(self, som):
        labels_qa = {}
        labels_id = {}
        for i in range(som.m):
            for j in range(som.n):
                # def label_represent_qa(self, question_weight, answer_weight, questions, answers, e, alpha):
                labels_id[(i, j)] = LabelSOM().label_represent_qa(som.questions_weights[i], som.answers_weights[j],
                                                                  preprocessing.scale(np.array(q_weight)),
                                                                  preprocessing.scale(np.array(a_weight)),
                                                                  som.qa[(i, j)], 0.005, 0.05)
                labels_qa[(i, j)] = list(np.array(som.qa[(i, j)])[labels_id[(i, j)]])
        print("代表性问答对。。。。。")
        print(som.qa)
        print(labels_id)
        print(labels_qa)

        # # 标记问题维度SOM
        x = {}
        y = {}
        for i in range(som.m):
            x[(i)] = LabelSOM().label_som(som.questions_weights[i], som.questions_dataitems[(i)], q_word, 0.005, 0.05)
        for j in range(som.n):
            y[(j)] = LabelSOM().label_som(som.answers_weights[j], som.answers_dataitems[(j)], a_word, 0.005, 0.001)
        print(x)
        print(y)

        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(7, 4))
        plt.xticks(list(np.arange(som.m + 2)))
        plt.yticks(list(np.arange(som.n + 2)))
        index2 = 0
        for x_index in range(som.m + 1):
            if x_index == 0:
                continue
            else:
                plt.text(x_index, index2 + 1, x[(x_index - 1)][0] + "\n" + x[(x_index - 1)][1])
        index1 = 0
        for y_index in range(som.n + 1):
            if y_index == 0:
                continue
            else:
                plt.text(index1, y_index + 1, y[(y_index - 1)][0] + "\n" + y[(y_index - 1)][1])
        for i in range(som.m + 1):
            for j in range(som.n + 1):
                if i == 0 or j == 0:
                    continue
                else:
                    plt.text(i, j + 0.5, str(len(labels_qa[(i - 1, j - 1)])))
        plt.grid(True)
        plt.show()

    def train(self):
        """
        GHSOM训练过程
        :return:
        """
        # 1.构建2*2的网络
        # (self, input_vects, m, n, iterations, alpha, neighborhood, layer, row_index, col_index)
        som = SOM.SOM(np.array(self.questions_input_vects),np.array(self.answers_input_vects), 2, 2, 1000, self.alpha, self.neighborhood, self.layer, None, None)
        # 训练SOM网络
        som.train()
        begin_train=0
        # 计算当前网络的量化均方误差
        q_MQE = som.questions_MQE
        a_MQE = som.answers_MQE
        weight=0.5
        print("初始化网络结构为：", som.m, "\t", som.n)
        print("初始化网络q量化误差：", som.questions_MQE)
        print("初始化网络a化误差：", som.answers_MQE)
        print("初始化网络聚类的结果：", "questions_labels:",
              som.questions_labels,"\n", "answers_labels:", som.answers_labels)
        if q_MQE >= self.tau_1 * self.questions_mqe0 or a_MQE >= self.tau_1 * self.answers_mqe0:
            if q_MQE >= self.tau_1 * self.questions_mqe0:
                print("问题方向SOM上：网络问题维度水平生长......")
                while q_MQE >= self.tau_1 * self.questions_mqe0:
                    print("问题维度水平生长。。。。。")
                    self.questions_horizontal_growing(som)
                    q_MQE = som.questions_MQE
                    print("问题维度量化误差为：", q_MQE)
                    print("网络结构：", som.m, "\t", som.n)
            if a_MQE >= self.tau_1 * self.answers_mqe0:
                print("答案方向SOM上：网络答案维度水平生长......")
                while a_MQE >= self.tau_1 * self.answers_mqe0:
                    print("答案维度水平生长。。。。。")
                    self.answers_horizontal_growing(som)
                    a_MQE = som.answers_MQE
                    print("答案维度量化误差为：", a_MQE)
                    print("网络结构：", som.m, "\t", som.n)
        print("第一层网络结构为：", som.m, "\t", som.n)
        print("第一层网络q量化误差：", q_MQE)
        print("第一层网络a化误差：", a_MQE)
        print("网络聚类的结果：", "questions_labels:",
              som.questions_labels, "\n", "answers_labels:", som.answers_labels)

        # 模型写入文件
        from sklearn.externals import joblib
        joblib.dump(som, "../data_output/net/som.model",compress=3)


        som_q_son_list=[]
        som_a_son_list=[]
        som_a_son_list.append(som)
        som_q_son_list.append(som)

        data_number = len(som.questions_input_vects)
        maps_representQAs=100
        # 计算当前网络的量化均方误差
        while begin_train==0 or (len(som_a_son_list)>0 or len(som_q_son_list)>0) :
            if self.layer==2:
                break
            if begin_train==0:
                print('GHSOM:第', self.layer,'层', q_MQE, self.tau_1 * self.questions_mqe0, a_MQE,
                  self.tau_1 * self.answers_mqe0)

            a_layer_num=self.layer
            q_layer_num=a_layer_num
            #print('答案维度神经元需垂直生长个数：', len(som_a_son_list))
            length_a = len(som_a_son_list)
            n_a = 0
            while len(som_a_son_list)>0 :
                som=som_a_son_list[0]
                q_MQE = som .questions_MQE
                a_MQE = som .answers_MQE

                print("结果写入文件中。。。。。。")
                size=0
                for key in som.questions_labels.keys():
                    size+=len(som.questions_labels[key])
                data_bum_questions = [-1]*size
                data_bum_answers = [-1]*size
                for key in som.questions_labels.keys():
                    for index in som.questions_labels[key]:
                        data_bum_questions[index] = key

                for key in som.answers_labels.keys():
                    for index in som.answers_labels[key]:
                        data_bum_answers[index] = key
                pd.DataFrame(data_bum_questions).to_excel("../data_output/"+str(self.layer)+"result_questions.xlsx",encoding="utf-8")
                pd.DataFrame(data_bum_answers).to_excel("../data_output/"+str(self.layer)+"result_answers.xlsx",encoding="utf-8")
                # with open("../data_output/"+str(self.layer)+"topic_words.txt","w",encoding="utf-8")as f:
                #     f.write(str(x)+"\n"+str(y))
                with open("../data_output/"+str(self.layer)+"result_questions.txt","w",encoding="utf-8")as f:
                    f.write("水平生长完成后网络结构为："+str(som.m)+"\t"+str(som.n)+"\n")
                    f.write("水平生长完成后网络q量化误差："+ str(q_MQE))
                    f.write("水平生长完成后网络a化误差："+ str(a_MQE)+"\n")

                with open("../data_output/"+str(self.layer)+"result_answers.txt","w",encoding="utf-8")as f:
                    f.write("参数为：tau1:"+str(self.tau_1)+"\t"+"tau2:"+str(self.tau_2))
                    f.write("水平生长完成后网络结构为："+str(som.m)+"\t"+str(som.n)+"\n")
                    f.write("水平生长完成后网络q量化误差："+ str(q_MQE))
                    f.write("水平生长完成后网络a化误差："+ str(a_MQE)+"\n")

                for i in range(len(data_bum_questions)):
                    with open("../data_output/"+str(self.layer)+"result_questions.txt", "a+", encoding="utf-8") as f:
                        f.write(str(data_bum_questions[i]) + "\n")

                    with open("../data_output/"+str(self.layer)+"result_answers.txt", "a+", encoding="utf-8") as f:
                        f.write(str(data_bum_answers[i]) + "\n")
                print("结果写入文件结束。。。。。")

                # 标记问题维度SOM
                print("labels.........")
                x = {}
                y = {}
                for i in range(som.m):
                    x[(i)] = LabelSOM.LabelSOM().label_som(som.questions_weights[i], som.questions_dataitems[(i)], self.q_word, 0.01,
                                          0.001)

                for j in range(som.n):
                    y[(j)] = LabelSOM.LabelSOM().label_som(som.answers_weights[j], som.answers_dataitems[(j)], self.a_word, 0.01, 0.001)
                # print("layer:" + str(self.layer) + "questions_answers:\n")
                print("questions_labels:\n", x)
                print("answers_labels:\n", y)
                x_df=pd.DataFrame.from_dict(list(x.items()))
                x_df.to_excel("../data_output/"+str(self.layer)+"result_questions(topic_words).xlsx",encoding="utf-8")
                y_df=pd.DataFrame.from_dict(list(y.items()))
                y_df.to_excel("../data_output/"+str(self.layer)+"result_answers(topic_words).xlsx",encoding="utf-8")
                represent_qas={}

                for row in range(som.m):
                    for col in range(som.n):
                        size=len(som.questions_input_vects)
                        questions_input_vects = som.questions_input_vects[som.qa[row, col]]
                        ansers_input_vects = som.answers_input_vects[som.qa[row, col]]
                        rate = len(som.qa[row, col]) / data_number
                        num_extr = np.ceil(rate * maps_representQAs)
                        represent_qas[row,col]=RepresentQAs.RepresentQAS().extraRepresentQAs(questions_input_vects,
                                                                                             ansers_input_vects,
                                                                                             num_extr,
                                                                                             0.5,
                                                                                             0.001)
                print("repreqas:\n",represent_qas)
                qa_df=pd.DataFrame.from_dict(list(represent_qas.items()))
                qa_df.to_excel("../data_output/"+str(self.layer)+"questions_answers(representQAs).xlsx",encoding="utf-8")
                #print(x)
                #print(y)
                #print("每一个聚类簇的结果。。。。。。。")
                #print("问题维度聚类簇。。。。。")
                #print(som.questions_labels)
                #print("答案维度聚类簇。。。。。")
                #print(som.answers_labels)
                # print("网络垂直生长......")
                # 水平生长完成，逐一检查是否需要进行垂直生长
                #问题维度垂直生长网络的集合
                question_vertical_growing_somset={}
                #答案维度垂直生长网络的集合
                answer_vertical_growing_somset = {}
                #问答垂直生长的网络集合
                qa_vertial_growing_somset={}
                #需要生长的问题神经元编号集合
                need_growing_questions_set=[]
                #需要生长的答案神经元编号集合
                need_growing_answers_set=[]
                for i in range(som.m):#问题维度垂直生长
                    if som.questions_mqe[i] >= self.tau_2 * self.questions_mqe0 and som.questions_dataitems_count[i]>2 :
                        print("需要向下生长的问题神经元为：",str(i))
                        print("问题方向SOM上：问题维度网络垂直生长......")
                        #print("当前问题维度神经元量化误差为：",som.questions_mqe[i])
                        question_vertical_growing_som=self.questions_vertical_growing(som,i)
                        question_vertical_growing_somset[(i)]=question_vertical_growing_som
                        som_q_son_list.append(question_vertical_growing_som)
                        need_growing_questions_set.append(i)
                        print("问题垂直生长完成后网络结构为：", question_vertical_growing_som.m, "\t", question_vertical_growing_som.n)
                        print("问题垂直生长完成后网络q量化误差：", question_vertical_growing_som.questions_MQE)
                        print("问题垂直生长完成后网络a化误差：", question_vertical_growing_som.answers_MQE)
                        print("网络聚类的结果：", "questions_labels:",
                              question_vertical_growing_som.questions_labels,"\n", "answers_labels:", question_vertical_growing_som.answers_labels)
                        qv_x = {}
                        qv_y = {}
                        for index_x in range(question_vertical_growing_som.m):
                            if len(question_vertical_growing_som.questions_dataitems[(index_x)])>0:
                                qv_x[(index_x)] = LabelSOM.LabelSOM().label_som(question_vertical_growing_som.questions_weights[index_x],
                                                                   question_vertical_growing_som.questions_dataitems[(index_x)], self.q_word, 0.01,
                                                                   0.001)
                            else:
                                qv_x[index_x]="当前神经元没有数据集"

                        for index_y in range(question_vertical_growing_som.n):
                            if len(question_vertical_growing_som.answers_dataitems[index_y])>0:
                                qv_y[(index_y)] = LabelSOM.LabelSOM().label_som(question_vertical_growing_som.answers_weights[index_y], question_vertical_growing_som.answers_dataitems[(index_y)],
                                                                   self.a_word, 0.01, 0.001)
                            else:
                                qv_y[index_y]="当前神经元没有数据集"

                        print("layer:" + str(self.layer) + str(i) + "next_questions:\n")
                        print("questions_labels:\n", qv_x)
                        print("answers_labels:\n", qv_y)
                            #print("垂直生长完成后网络的q量化误差为：",question_vertical_growing_som.questions_MQE)
                       #print("是否还需要垂直生长？", self.is_need_questions_vertical_growing(question_vertical_growing_som))
                        size = 0
                        for key in question_vertical_growing_som.questions_labels.keys():
                            size += len(question_vertical_growing_som.questions_labels[key])
                        data_bum_questions = [-1]*size
                        data_bum_answers = [-1]*size
                        for key in question_vertical_growing_som.questions_labels.keys():
                            for index in question_vertical_growing_som.questions_labels[key]:
                                data_bum_questions[index] = key

                        for key in question_vertical_growing_som.answers_labels.keys():
                            for index in question_vertical_growing_som.answers_labels[key]:
                                data_bum_answers[index] = key
                        print("i:",i)
                        pd.DataFrame(data_bum_questions).to_excel(
                            "../data_output/" + "layer"+str(self.layer)+str(i) + "next_questions.xlsx", encoding="utf-8")
                        pd.DataFrame(data_bum_answers).to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(i) + "questions(answers).xlsx", encoding="utf-8")
                        x_df = pd.DataFrame.from_dict(list(qv_x.items()))
                        x_df.to_excel(
                            "../data_output/" + "layer"+str(self.layer)+str(i) + "next_questions(topic_words).xlsx", encoding="utf-8")

                        y_df = pd.DataFrame.from_dict(list(qv_y.items()))
                        y_df.to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(i) + "questions(answers)(topic_words).xlsx", encoding="utf-8")
                        represent_qas = {}
                        for row in range(question_vertical_growing_som.m):
                            for col in range(question_vertical_growing_som.n):
                                size=len(question_vertical_growing_som.questions_input_vects)
                                questions_input_vects = question_vertical_growing_som.questions_input_vects[question_vertical_growing_som.qa[row, col]]
                                ansers_input_vects = question_vertical_growing_som.answers_input_vects[question_vertical_growing_som.qa[row, col]]
                                rate = len(question_vertical_growing_som.qa[row, col]) / data_number
                                num_extr = np.ceil(rate * maps_representQAs)
                                represent_qas[row, col] = RepresentQAs.RepresentQAS().extraRepresentQAs(
                                    questions_input_vects,
                                    ansers_input_vects,
                                    num_extr,
                                    0.5,
                                    0.001)
                        print("repreqas:\n", represent_qas)
                        qa_df = pd.DataFrame.from_dict(list(represent_qas.items()))
                        qa_df.to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(i) + "questions(representQAs).xlsx", encoding="utf-8")
                        joblib.dump(question_vertical_growing_som,"../data_output/net/"+"questions_cluster_growing"+str(i)+".model",compress=3)
                        # with open("../data_output/" + "layer"+str(self.layer)+str(i) + "next_questions(topic_words).txt","w", encoding="utf-8")as f:
                        #     f.write(str(x)+"\n"+str(y))
                        # with open("../data_output/" + "layer:"+str(self.layer)+str(i) + "next_questions.txt", "w", encoding="utf-8")as f:
                        #     f.write("水平生长完成后网络结构为：" + str(question_vertical_growing_som.m)
                        #             + "\t" + str(question_vertical_growing_som.n) + "\n")
                        #     f.write("水平生长完成后网络q量化误差：" + str(question_vertical_growing_som.questions_MQE))
                        #     f.write("水平生长完成后网络a化误差：" + str(question_vertical_growing_som.answers_MQE)+"\n")
                        #
                        # with open("../data_output/" + "layer:"+str(self.layer)+ str(i) + "questions(answers).txt", "w", encoding="utf-8")as f:
                        #     f.write("参数为：tau1:" + str(self.tau_1) + "\t" + "tau2:" + str(self.tau_2))
                        #     f.write("水平生长完成后网络结构为：" + str(question_vertical_growing_som.m)
                        #             + "\t" + str(question_vertical_growing_som.n) + "\n")
                        #     f.write("水平生长完成后网络q量化误差：" + str(question_vertical_growing_som.questions_MQE))
                        #     f.write("水平生长完成后网络a化误差：" + str(question_vertical_growing_som.answers_MQE)+"\n")
                        #
                        # for i in range(len(data_bum_questions)):
                        #     with open("../data_output/" + "layer:"+str(self.layer)+ str(i) + "next_questions.txt", "a+",
                        #               encoding="utf-8") as f:
                        #         f.write(str(data_bum_questions[i]) + "\n")
                        #
                        #     with open("../data_output/" + "layer:"+str(self.layer)+ str(i) +"questions(answers).txt", "a+",
                        #               encoding="utf-8") as f:
                        #         f.write(str(data_bum_answers[i]) + "\n")
                for j in range(som.n):#答案维度垂直生长
                    #print("答案维度网络垂直生长......")
                    if som.answers_mqe[j] >= self.tau_2 * self.answers_mqe0 and som.answers_dataitems_count[j]>2:
                        print("答案方向SOM上：答案维度网络垂直生长......")
                        print("答案维度上需要垂直生长的神经元：",str(j))
                        #print("当前答案维度神经元量化误差为：", som.answers_mqe[j])
                        answer_vertical_growing_som=self.answers_vertical_growing(som,j)
                        answer_vertical_growing_somset[(j)]=answer_vertical_growing_som
                        som_a_son_list.append(answer_vertical_growing_som)
                        need_growing_answers_set.append(j)
                        print("答案垂直生长完成后网络结构为：", answer_vertical_growing_som.m, "\t", answer_vertical_growing_som.n)
                        print("答案垂直生长完成后网络q量化误差：", answer_vertical_growing_som.questions_MQE)
                        print("答案垂直生长完成后网络a化误差：", answer_vertical_growing_som.answers_MQE)
                        print("网络聚类的结果：", "questions_labels:",
                              answer_vertical_growing_som.questions_labels,"\n", "answers_labels:", answer_vertical_growing_som.answers_labels)
                        av_x = {}
                        av_y = {}
                        for index_x in range(answer_vertical_growing_som.m):
                            if len(answer_vertical_growing_som.questions_dataitems[index_x])>0:
                                av_x[(index_x)] = LabelSOM.LabelSOM().label_som(answer_vertical_growing_som.questions_weights[index_x],
                                                                   answer_vertical_growing_som.questions_dataitems[
                                                                       (index_x)], self.q_word, 0.01,
                                                                   0.001)
                            else:
                                av_x[index_x]="当前神经元没有数据集"

                        for index_y in range(answer_vertical_growing_som.n):
                            if len(answer_vertical_growing_som.answers_dataitems[index_y])>0:
                                av_y[(index_y)] = LabelSOM.LabelSOM().label_som(answer_vertical_growing_som.answers_weights[index_y],
                                                                   answer_vertical_growing_som.answers_dataitems[(index_y)],
                                                                   self.a_word, 0.01, 0.001)
                            else:
                                av_y[index_y]="当前神经元没有数据集"
                        print("layer:"+str(self.layer)+ str(j) + "next_answers:\n")
                        print("questions_labels:\n",av_x)
                        print("answers_labels:\n",av_y)
                        size = 0
                        for key in answer_vertical_growing_som.questions_labels.keys():
                            size += len(answer_vertical_growing_som.questions_labels[key])
                        data_bum_questions = [-1]*size
                        data_bum_answers = [-1]*size
                        for key in answer_vertical_growing_som.questions_labels.keys():
                            for index in answer_vertical_growing_som.questions_labels[key]:
                                data_bum_questions[index] = key

                        for key in answer_vertical_growing_som.answers_labels.keys():
                            for index in answer_vertical_growing_som.answers_labels[key]:
                                data_bum_answers[index] = key
                        print("j:",j)
                        pd.DataFrame(data_bum_questions).to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(j) + "answers(questions).xlsx",
                            encoding="utf-8")
                        pd.DataFrame(data_bum_answers).to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(j) +"next_answers.xlsx",
                            encoding="utf-8")
                        x_df = pd.DataFrame.from_dict(list(av_x.items()))
                        x_df.to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(j) + "answers(questions)(topic_words).xlsx",
                            encoding="utf-8")

                        y_df = pd.DataFrame.from_dict(list(av_y.items()))
                        y_df.to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(j) +"next_answers(topic_words).xlsx",
                            encoding="utf-8")
                        represent_qas = {}
                        for row in range(answer_vertical_growing_som.m):
                            for col in range(answer_vertical_growing_som.n):
                                size=len(answer_vertical_growing_som.questions_input_vects)
                                questions_input_vects = answer_vertical_growing_som.questions_input_vects[answer_vertical_growing_som.qa[row, col]]
                                ansers_input_vects = answer_vertical_growing_som.answers_input_vects[answer_vertical_growing_som.qa[row, col]]
                                rate = len(answer_vertical_growing_som.qa[row, col]) / data_number
                                num_extr = np.ceil(rate * maps_representQAs)
                                represent_qas[row, col] = RepresentQAs.RepresentQAS().extraRepresentQAs(
                                    questions_input_vects,
                                    ansers_input_vects,
                                    num_extr,
                                    0.5,
                                    0.001)
                        print("repreqas:\n", represent_qas)
                        qa_df = pd.DataFrame.from_dict(list(represent_qas.items()))
                        qa_df.to_excel(
                            "../data_output/" + "layer"+str(self.layer)+ str(j) +"answers(representQAs).xlsx",
                            encoding="utf-8")
                        joblib.dump(answer_vertical_growing_som,
                                    "../data_output/net/" + "answers_cluster_growing" + str(j) + ".model",compress=3)
                        # with open("../data_output/" + "layer:" + str(self.layer) + str(j) + "next_answers(topic_words).txt", "w",
                        #           encoding="utf-8")as f:
                        #     f.write(str(x)+"\n"+str(j))


                        # with open("../data_output/" + "layer:"+str(self.layer)+ str(j) + "next_answers.txt", "w", encoding="utf-8")as f:
                        #     f.write("答案维度向下生长的神经元为："+str(som.n)+"\n")
                        #     f.write("生长完成后网络结构为：" + str(answer_vertical_growing_som.m)
                        #             + "\t" + str(answer_vertical_growing_som.n) + "\n")
                        #     f.write("生长完成后网络q量化误差：" + str(answer_vertical_growing_som.questions_MQE))
                        #     f.write("生长完成后网络a化误差：" + str(answer_vertical_growing_som.answers_MQE)+"\n")
                        #
                        # with open("../data_output/" + "layer:"+str(self.layer)+ str(j) +"answers(questions).txt", "w", encoding="utf-8")as f:
                        #     f.write("答案维度向下生长的神经元为：" + str(som.n) + "\n")
                        #     f.write("参数为：tau1:" + str(self.tau_1) + "\t" + "tau2:" + str(self.tau_2))
                        #     f.write("生长完成后网络结构为：" + str(answer_vertical_growing_som.m)
                        #             + "\t" + str(answer_vertical_growing_som.n) + "\n")
                        #     f.write("生长完成后网络q量化误差：" + str(answer_vertical_growing_som.questions_MQE))
                        #     f.write("生长完成后网络a化误差：" + str(answer_vertical_growing_som.answers_MQE)+"\n")
                        #
                        # for i in range(len(data_bum_questions)):
                        #     with open("../data_output/" + "layer:"+str(self.layer)+str(j) + "next_answers.txt", "a+",
                        #               encoding="utf-8") as f:
                        #         f.write(str(data_bum_questions[i]) + "\n")
                        #
                        #     with open("../data_output/" + "layer:"+str(self.layer)+ str(j) + "answers(questions).txt", "a+",
                        #               encoding="utf-8") as f:
                        #         f.write(str(data_bum_answers[i]) + "\n")

                        #print("垂直生长完成后网络的a量化误差为：", question_vertical_growing_som.answers_MQE)
                        #print("是否还需要垂直生长？", self.is_need_answers_vertical_growing(answer_vertical_growing_som))

                for i in need_growing_questions_set:
                    for j in need_growing_answers_set:
                        if len(som.qa[i, j]) > 4 and (1-weight)*som.questions_mqe[i]+weight*som.answers_mqe[j]>self.tau_2 * self.questions_mqe0+self.tau_2 * self.answers_mqe0:
                            print("qa的生长。。。")
                            qa_som=self.qa_vertical_growing_som(som,i,j,question_vertical_growing_somset[i].questions_weights,answer_vertical_growing_somset[j].answers_weights)
                            joblib.dump(qa_som,"../data_output/net/" + "qa_cluster_growing" + str(i)+"," +str(j)+ ".model",compress=3)
                if begin_train==0:
                    som_a_son_list.remove(som_a_son_list[0])
                    som_q_son_list.remove(som_q_son_list[0])
                else:
                    som_a_son_list.remove(som_a_son_list[0])
                n_a=n_a+1
                print('GHSOM:第', a_layer_num, '层', '答案维度神经元生长进度：', str(n_a), '/', length_a)
                if n_a==length_a:
                    break


            print('GHSOM:从问题开始第',q_layer_num , '层', q_MQE, self.tau_1 * self.questions_mqe0, a_MQE,
                  self.tau_1 * self.answers_mqe0)

            print('GHSOM:第',q_layer_num , '层', '问题维度神经元需垂直生长个数：',len(som_q_son_list))
            length_q=len(som_q_son_list)
            n_q=0
            while len(som_q_son_list) > 0:

                som=som_q_son_list[0]
                q_MQE =som .questions_MQE
                a_MQE = som .answers_MQE
                #print('GHSOM', begin_train, som.n, som.m, self.layer, q_MQE, self.tau_1 * self.questions_mqe0,a_MQE, self.tau_1 * self.answers_mqe0)
                #print("q量化均方误差为：", q_MQE)
                #print("a量化均方误差为：", a_MQE)
                #print("questions_mqe0为：", self.questions_mqe0)
                #print("answers_mqe0为：", self.answers_mqe0)

                # # 标记问题维度SOM
                #print("labels.........")
                x = {}
                y = {}
                for i in range(som.m):
                    x[(i)] = LabelSOM.LabelSOM().label_som(som.questions_weights[i], som.questions_dataitems[(i)],
                                                               self.q_word, 0.01,
                                                               0.001)
                for j in range(som.n):
                    y[(j)] = LabelSOM.LabelSOM().label_som(som.answers_weights[j], som.answers_dataitems[(j)],
                                                             self.a_word, 0.01, 0.001)

                print("questions_labels_topic:\n",x)
                print("answers_labels_topic:\n",y)
                #print("每一个聚类簇的结果。。。。。。。")
                #print("问题维度聚类簇。。。。。")
                #print(som.questions_labels)
                #print("答案维度聚类簇。。。。。")
                #print(som.answers_labels)
                #水平生长完成，逐一检查是否需要进行垂直生长
                #问题维度垂直生长网络的集合
                question_vertical_growing_somset = {}
                # 答案维度垂直生长网络的集合
                answer_vertical_growing_somset = {}
                # 需要生长的问题神经元编号集合
                need_growing_questions_set = []
                # 需要生长的答案神经元编号集合
                need_growing_answers_set = []
                for i in range(som.m):  # 问题维度垂直生长
                    if som.questions_mqe[i] >= self.tau_2 * self.questions_mqe0 and som.questions_dataitems_count[i]>2:
                        print("问题方向SOM上：问题维度网络垂直生长......")
                        # print("当前问题维度神经元量化误差为：",som.questions_mqe[i])
                        question_vertical_growing_som = self.questions_vertical_growing(som, i)
                        question_vertical_growing_somset[(i)] = question_vertical_growing_som
                        som_q_son_list.append(question_vertical_growing_som)
                        need_growing_questions_set.append(i)
                        print("问题垂直生长完成后网络结构为：", question_vertical_growing_som.m, "\t", question_vertical_growing_som.n)
                        print("问题垂直生长完成后网络q量化误差：", question_vertical_growing_som.questions_MQE)
                        print("问题垂直生长完成后网络a化误差：", question_vertical_growing_som.answers_MQE)
                        print("网络聚类的结果：", "questions_labels:",
                              question_vertical_growing_som.questions_labels,"\n", "answers_labels:",
                              question_vertical_growing_som.answers_labels)
                        x = {}
                        y = {}
                        for i in range(question_vertical_growing_som.m):
                            if len(question_vertical_growing_som.questions_dataitems[i])>0:
                                x[(i)] = LabelSOM.LabelSOM().label_som(question_vertical_growing_som.questions_weights[i],
                                                                   question_vertical_growing_som.questions_dataitems[(i)],
                                                                   self.q_word, 0.01,
                                                                   0.001)
                            else:
                                x[i]="当前神经元没有数据集"
                        for j in range(question_vertical_growing_som.n):
                            if len(question_vertical_growing_som.answers_dataitems[j])>0:
                                y[(j)] = LabelSOM.LabelSOM().label_som(question_vertical_growing_som.answers_weights[j], question_vertical_growing_som.answers_dataitems[(j)],
                                                                   self.a_word, 0.01, 0.001)
                            else:
                                y[j]="当前神经元没有数据集"
                        print("ver(q)_questions_labels_topic:\n",x)
                        print("ver(q)_answers_labels_topic:\n",y)
                        #print("垂直生长完成后网络的q量化误差为：", question_vertical_growing_som.questions_MQE)
                        #print("是否还需要垂直生长？", self.is_need_questions_vertical_growing(question_vertical_growing_som))

                for j in range(som.n):  # 答案维度垂直生长
                    #print("答案维度网络垂直生长......")
                    if som.answers_mqe[j] >= self.tau_2 * self.answers_mqe0 and som.answers_dataitems_count[j]>2:
                        print("答案方向SOM上：答案维度网络垂直生长......")
                        # print("当前答案维度神经元量化误差为：", som.answers_mqe[j])
                        answer_vertical_growing_som = self.answers_vertical_growing(som, j)
                        answer_vertical_growing_somset[(j)] = answer_vertical_growing_som
                        som_a_son_list.append(answer_vertical_growing_som)
                        need_growing_answers_set.append(j)
                        print("问题垂直生长完成后网络结构为：", answer_vertical_growing_som.m, "\t", answer_vertical_growing_som.n)
                        print("问题垂直生长完成后网络q量化误差：", answer_vertical_growing_som.questions_MQE)
                        print("问题垂直生长完成后网络a化误差：", answer_vertical_growing_som.answers_MQE)
                        print("网络聚类的结果：", "questions_labels:",
                              answer_vertical_growing_som.questions_labels,"\n", "answers_labels:",
                              answer_vertical_growing_som.answers_labels)
                        x = {}
                        y = {}
                        for i in range(answer_vertical_growing_som.m):
                            if len(answer_vertical_growing_som.questions_dataitems[i])>0:
                                x[(i)] = LabelSOM.LabelSOM().label_som(answer_vertical_growing_som.questions_weights[i],
                                                                   answer_vertical_growing_som.questions_dataitems[
                                                                       (i)],
                                                                   self.q_word, 0.01,
                                                                   0.001)
                            else:
                                x[i]="当前神经元没有数据集"
                        for j in range(answer_vertical_growing_som.n):
                            if len(answer_vertical_growing_som.answers_dataitems[j])>0:
                              y[(j)] = LabelSOM.LabelSOM().label_som(answer_vertical_growing_som.answers_weights[j],
                                                                   answer_vertical_growing_som.answers_dataitems[(j)],
                                                                   self.a_word, 0.01, 0.001)
                            else:
                                y[j]="当前神经元没有数据集"
                        print("ver(a)_questions_labels_topic:\n", x)
                        print("ver(a)_answers_labels_topic:\n", y)

                        # print("垂直生长完成后网络的a量化误差为：", question_vertical_growing_som.answers_MQE)
                        #print("是否还需要垂直生长？", self.is_need_answers_vertical_growing(answer_vertical_growing_som))

                for i in need_growing_questions_set:
                    for j in need_growing_answers_set:
                        if len(som.qa[i, j]) > 4 and (1-weight)*som.questions_mqe[i]+weight*som.answers_mqe[j]>self.tau_2 * self.questions_mqe0+self.tau_2 * self.answers_mqe0:
                            print("qa的生长。。。")
                            qa_som = self.qa_vertical_growing_som(som, i, j,
                                                                  question_vertical_growing_somset[i].questions_weights,
                                                                  answer_vertical_growing_somset[j].answers_weights)
                            joblib.dump(qa_som,
                                        "../data_output/net/" + "qa_cluster_growing" + str(i) + "," + str(j) + ".model",compress=3)

                som_q_son_list.remove(som_q_son_list[0])
                n_q=n_q+1
                print('GHSOM:第', q_layer_num, '层', '问题维度神经元生长进度：',str(n_q),'/',length_q)
                if n_q==length_q:
                    break
            self.layer+=1
            begin_train=1


if __name__=="__main__":
    data_path = "../data/782个问答对/"
    # import pandas as pd
    # questions=pd.read_excel(data_path+"questions_100topics.xlsx",encoding="utf-8")
    # answers   =pd.read_excel(data_path+"answers_100topics.xlsx",encoding="utf-8")
    import pickle
    from sklearn import preprocessing
    q_weight_pkl_file = open(
        data_path+'q_weight.pkl', 'rb')
    q_weight = pickle.load(q_weight_pkl_file)
    a_weight_pkl_file = open(
        data_path+'a_weight.pkl', 'rb')
    a_weight = pickle.load(a_weight_pkl_file)
    q_word_pkl_file = open(data_path+'q_word.pkl', 'rb')
    q_word = pickle.load(q_word_pkl_file)
    a_word_pkl_file = open(data_path+'a_word.pkl', 'rb')
    a_word = pickle.load(a_word_pkl_file)
    print(q_weight)
    print(np.array(q_weight).shape)
    print(a_weight)
    print(np.array(a_weight).shape)
    from sklearn import preprocessing
    print("使用cos来衡量相似性的GHSOM")
    input_vects_questions=q_weight
    input_vects_answers=a_weight
    ghsom = GHSOM(input_vects_questions, input_vects_answers, 0.08, 0.05, "gaussian", 0.3,q_word,a_word)
    ghsom.train()
    print("网络的层数为：", ghsom.layer)