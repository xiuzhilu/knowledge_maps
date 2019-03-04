import numpy as np
class SOM(object):

    #QA_SOM网络的初始化
    def __init__(self,questions_input_vects,answers_input_vects,m, n, iterations, alpha, neighborhood,layer,row_index,col_index):
        #preprocessing.scale(np.array(q_weight)), preprocessing.scale(np.array(
            #a_weight)), m = 7, n = 12, iterations = 1, alpha = 0.3, neighborhood = "gaussian", layer = 1, row_index = None, col_index = None
        """
        :param questions_input_vects: 问题维度输入数据
        :param answers_input_vects: 答案维度输入数据
        :param m: 问题维度神经元
        :param n: 答案维度神经元
        :param iterations: 迭代的次数
        :param alpha: 学习率
        :param neighborhood:领域
        :param layer: 网络所处的层级
        :param row_index: 上一层中问题被扩展的神经元编号，若是第一层则为None
        :param col_index: 上一层中答案被扩展的神经元编号，若是第一层则为None
        """
        # np.random.seed(0)
        self.questions_input_vects=questions_input_vects
        self.answers_input_vects=answers_input_vects
        self.m = m
        self.n = n
        self.iterations = iterations
        self.alpha0 = alpha
        self.alpha = alpha
        self.layer = layer
        self.row_index = row_index
        self.col_index = col_index
        if neighborhood is None:
            self.neighborhood = "gaussian"
        else:
            self.neighborhood = neighborhood
        questions_weight=[]
        self.questions_labels={}
        self.questions_dataitems = {}
        self.questions_dataitems_count = {}
        for i in range(self.m):
            questions_weight.append(np.random.random(self.questions_input_vects.shape[1]))
            self.questions_dataitems[(i)]=[]
            self.questions_dataitems_count[(i)]=0
            self.questions_labels[(i)]=[]
        self.questions_weights=np.array(questions_weight)
        answers_weight=[]
        self.answers_labels={}
        self.answers_dataitems = {}
        self.answers_dataitems_count = {}
        for j in range(self.n):
            answers_weight.append(np.random.random(self.answers_input_vects.shape[1]))
            self.answers_dataitems[(j)] = []
            self.answers_dataitems_count[(j)] = 0
            self.answers_labels[(j)]=[]
        self.answers_weights=np.array(answers_weight)
        # 网络的量化均方误差
        self.questions_MQE = 0
        self.answers_MQE = 0
        self.qa_MQE=0.5*self.questions_MQE+0.5*self.answers_MQE
        # 初始化网络的每个神经元的平均量化误差
        self.questions_mqe = np.zeros(self.m)
        self.answers_mqe = np.zeros(self.n)
        self.qa={}
        self.qa_count=np.zeros([self.m,self.n])
        for i in range(self.m):
            for j in range(self.n):
                self.qa[(i,j)]=[]

    def getquestion_winner_unit(self,question_input_vect):
        """
        找到问题维度的获胜神经元编号并且返回
        :param question_input_vect: 当前问题维度的输入数据
        :return: 问题维度获胜神经元编号
        """
        row=0
        dist=np.float("-inf")
        for i in range(self.m):
            d=self.get_distance(question_input_vect,self.questions_weights[i])
            if d>dist:
                row=i
                dist=d
        return row

    def getanswer_winner_unit(self,answer_input_vect):
        """
        找到答案维度的获胜神经元编号并返回
        :param answer_input_vect: 当前答案维度的输入数据
        :return:答案维度获胜神经元编号
        """
        col=0
        dist=np.float("-inf")
        for j in range(self.n):
            d=self.get_distance(answer_input_vect,self.answers_weights[j])
            if d>dist:
                col=j
                dist=d
        return col

    def KL(self, P, Q):
        addsum = 0
        for i in range(len(P)):
            try:
                sum = float(P[i]) * np.log(float(P[i]) / float(Q[i]))
            except:
                sum = 0
            addsum = addsum + sum
        return addsum

    # JS
    def JS(self, dataA, dataB):
        dataC = []
        # print(len(dataB),len(dataA),'----------------------------')
        for i in range(len(dataA)):
            data3 = (dataA[i] + dataB[i]) * 1.0 / 2
            dataC.append(data3)
        JSdata = 0.5 * self.KL(dataA, dataC) + 0.5 * self.KL(dataB, dataC)
        return JSdata

    # print(W_C_Q)
    # sim=(max-js)/(max-min)
    def sim(self, P, Q):
        sim_data = 1 - self.JS(P, Q)
        return sim_data

    def get_distance(self, input_vect, weight):
        """
        计算样本与神经元权重的距离
        :param input_vect: 样本
        :param weight: 权重
        :return: 距离
        """
        # 余弦相似度计算
        dist=np.sum(np.array(input_vect)*np.array(weight))/(np.sqrt(np.sum(np.array(input_vect)**2))*np.sqrt(np.sum(np.array(weight)**2)))
        #dist = np.sqrt(np.sum(np.power(input_vect - weight, 2)))#欧式距离
        # JS相似度计算
        #dist = self.JS(input_vect, weight)
        return dist

    def update_alpha(self):
        """
        更新学习率
        :param iteration: 迭代的次数
        :return: 无返回值
        """
        self.alpha -= self.alpha0 / self.iterations
        self.alpha = max(self.alpha, 0.0001)

    def neighborhood_calculate_questions(self,neighborhood,winner_question,i, radius):
        """
        :param neighborhood: 领域类型gaussian
        :param winner_question: 问题维度获胜神经
        :param i: 要更新的神经元编号
        :param radius: 领域半径
        :return: 高斯领域
        """
        #计算要更新的神经元与获胜神经元的距离
        dist=self.get_distance(self.questions_weights[winner_question],self.questions_weights[i])
        #dist=np.power(np.abs(winner_row-i),2)+np.power(np.abs(winner_col-j),2)
        if neighborhood=="gaussian":
            return np.power(np.e,-1.0*dist**2/(2.0*radius**2))#np.exp(-1.0*dist/(2.0*radius**2))

    def neighborhood_calculate_answers(self,neighborhood, winner_answer, j, radius):
        """

        :param neighborhood: 领域
        :param winner_answer: 答案维度获胜神经元
        :param j: 答案神经元
        :param radius: 领域半径
        :return:
        """
        # 计算要更新的神经元与获胜神经元的距离
        dist = self.get_distance(self.answers_weights[winner_answer], self.answers_weights[j])
        # dist=np.power(np.abs(winner_row-i),2)+np.power(np.abs(winner_col-j),2)
        if neighborhood == "gaussian":
            return np.power(np.e,-1.0 * dist**2 / (2.0 * radius ** 2))#np.exp(-1.0 * dist / (2.0 * radius ** 2))

    def update_questions_weights(self,question_input_vect,winner_question,radius):
        """
        调整网络问题维度的权重
        :param question_input_vect: 当前问题维度的输入
        :param winner_question: 问题维度获胜神经元
        :param radiu: 领域半径
        :return:
        """
        #更新获胜神经元的权重
        self.questions_weights[winner_question]=self.questions_weights[winner_question]+self.alpha*(question_input_vect-self.questions_weights[winner_question])
        #更新获胜神经元领域权重
        for i in range(self.m):
            if i!=winner_question:
                h = self.neighborhood_calculate_questions(self.neighborhood, winner_question,i,radius)
                self.questions_weights[i]=self.questions_weights[i]+self.alpha*h*(question_input_vect-self.questions_weights[winner_question])

    def update_answers_weights(self,answer_input_vect,winner_answer,radius):
        """
        更新答案维度全红
        :param answer_input_vect:答案输入
        :param answer_question:答案维度获胜神经元
        :param radius:领域半径
        :return:
        """
        # 更新获胜神经元的权重
        self.answers_weights[winner_answer] = self.answers_weights[winner_answer] + self.alpha * (
            answer_input_vect - self.answers_weights[winner_answer])
        # 更新获胜神经元领域权重
        for j in range(self.n):
            if j != winner_answer:
                h = self.neighborhood_calculate_answers(self.neighborhood, winner_answer, j, radius)
                self.answers_weights[j] = self.answers_weights[j] + self.alpha*h* (
                    answer_input_vect - self.answers_weights[winner_answer])

    def calculate_questions_mqe(self,row):
        """

        :param row: 问题维度神经元编号
        :return:
        """
        if self.questions_dataitems.__contains__(row):
            if len(self.questions_dataitems[(row)])==0:
                self.questions_mqe[row]=0
            else:
                #mqe=np.mean(np.sqrt(np.sum(np.power(self.questions_dataitems[(row)] - self.questions_weights[row], 2),axis=1)))
                sum=0
                for i in range(len(self.questions_dataitems[(row)])):
                    sum+=1-self.get_distance(self.questions_dataitems[(row)][i],self.questions_weights[row])
                avg=np.mean(sum)
                self.questions_mqe[row]=avg

    def calculate_answers_mqe(self,col):
        """

        :param col: 答案维度神经元编号
        :return:
        """
        if self.answers_dataitems.__contains__(col):
            if len(self.answers_dataitems[(col)])==0:
                self.answers_mqe[col]=0
            else:
                #mqe=np.mean(np.sqrt(np.sum(np.power(self.answers_dataitems[(col)] - self.answers_weights[col], 2),axis=1)))
                sum=0
                for j in range(len(self.answers_dataitems[(col)])):
                    sum+=1-self.get_distance(self.answers_dataitems[(col)][j],self.answers_weights[col])
                avg=np.mean(sum)
                self.answers_mqe[col]=avg

    def calculate_qeustions_MQE(self):
        count = 0
        for i in range(self.m):
            if len(self.questions_dataitems[i]) == 0:#没有数据映射上去的神经元，不能计数，避免存在很多死神经元来平均整个网络的量化误差
                continue
            else:
                count += 1
        self.questions_MQE = np.sum(self.questions_mqe) / count

    def calculate_answers_MQE(self):
        count = 0
        for j in range(self.n):
            if len(self.answers_dataitems[j]) == 0:#没有数据映射上去的神经元，不能计数，避免存在很多死神经元来平均整个网络的量化误差
                continue
            else:
                count += 1
        self.answers_MQE = np.mean(self.answers_mqe) / count

    def calculate_qa_MQE(self):
        self.qa_MQE=0.5*self.questions_MQE+0.5*self.answers_MQE

    def get_max_question_mqe_index(self):
        """
        计算并返回问题维度量化均方误差最大的神经元编号
        :return: 问题维度量化均方误差最大的神经元编号
        """
        max_index=0
        max_mqe=np.float("-inf")
        for i in range(self.m):
            if self.questions_mqe[i]>max_mqe:
                max_index=i
                max_mqe=self.questions_mqe[i]
        return max_index

    def get_max_answer_mqe_index(self):
        """
        计算并返回答案维度量化均方误差最大的神经元编号
        :return: 答案维度量化均方误差最大的神经元编号
        """
        max_index = 0
        max_mqe = np.float("-inf")
        for j in range(self.n):
            if self.answers_mqe[j] > max_mqe:
                max_index = j
                max_mqe = self.answers_mqe[j]
        return max_index

    def projector_questions(self,questions_input_vects):
        for i in range(len(questions_input_vects)):
            question_input_vect = questions_input_vects[i]
            # 寻找问题维度的获胜神经元
            winner_question = self.getquestion_winner_unit(question_input_vect)
            #问题维度映射数据
            self.questions_dataitems[(winner_question)].append(question_input_vect)
            self.questions_dataitems_count[(winner_question)]+=1
            if self.questions_labels.__contains__(winner_question):
                self.questions_labels[(winner_question)].append(i)
            else:
                self.questions_labels[(winner_question)]=[i]

    def projector_answers(self,answers_input_vects):
        for i in range(len(answers_input_vects)):
            answer_input_vect = self.answers_input_vects[i]
            # 寻找答案维度的获胜神经元
            winner_answer = self.getanswer_winner_unit(answer_input_vect)
            #答案维度映射数据
            self.answers_dataitems[(winner_answer)].append(answer_input_vect)
            self.answers_dataitems_count[(winner_answer)]+=1
            if self.answers_labels.__contains__(winner_answer):
                self.answers_labels[(winner_answer)].append(i)
            else:
                self.answers_labels[(winner_answer)]=[i]

    def projector_qa(self,questions_input_vects,answers_input_vects):
        for i in range(len(questions_input_vects)):
            question_input_vect = questions_input_vects[i]
            answer_input_vect = answers_input_vects[i]
            # 寻找问题和答案维度的获胜神经元
            winner_question = self.getquestion_winner_unit(question_input_vect)
            winner_answer = self.getanswer_winner_unit(answer_input_vect)
            #qa维度映射数据
            self.qa[(winner_question,winner_answer)].append(i)#映射到神经元[winner_question,winner_answer]的qa对编号
            self.qa_count[winner_question,winner_answer]+=1

    def train(self):
        iteration = 0
        radius = np.linspace(1, self.iterations, self.iterations)
        #初始化self.iteration=1
        while iteration < self.iterations:
            print("当前的学习率为：", self.alpha, "当前迭代次数：", iteration)
            #print(len(self.questions_input_vects))
            for i in range(len(self.questions_input_vects)):
                question_input_vect=self.questions_input_vects[i]
                #print(question_input_vect)
                answer_input_vect=self.answers_input_vects[i]
                # 寻找问题维度的获胜神经元
                winner_question = self.getquestion_winner_unit(question_input_vect)
                #print(winner_question)
                #寻找答案维度的获胜神经元
                winner_answer=self.getanswer_winner_unit(answer_input_vect)
                #调整问题维度的权重
                self.update_questions_weights(question_input_vect,winner_question,radius[iteration])
                #调整答案维度的权重
                self.update_answers_weights(answer_input_vect, winner_answer, radius[iteration])
            iteration += 1
            self.update_alpha()
        #训练完成映射数据
        self.projector_questions(self.questions_input_vects)
        self.projector_answers(self.answers_input_vects)
        self.projector_qa(self.questions_input_vects,self.answers_input_vects)#映射qa
        #计算问题维度每个神经元的量化误差
        for i in range(self.m):
            self.calculate_questions_mqe(i)
        #计算答案维度每个神经元的量化误差
        for j in range(self.n):
            self.calculate_answers_mqe(j)
        #计算答案维度量化误差
        self.calculate_qeustions_MQE()
        #计算答案维度量化误差
        self.calculate_answers_MQE()
        #计算问答对的量化误差
        self.calculate_qa_MQE()