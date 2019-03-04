import numpy as np
class LabelSOM(object):
    def label_represent_qa(self,question_weight,answer_weight,questions,answers,qa_index,e,alpha):
        """
        标记qa对的位置
        :param question_weight: 问题维度神经元编号
        :param answer_weight: 答案维度神经元编号
        :param q_word: 问题维度词袋
        :param a_word: 答案维度词袋
        :param e: 一个足够小的正数
        :param alpha: 权重阈值
        :return:
        """
        L_qa=[]
        qa_sim=[]
        for i in qa_index:
            question=questions[i]
            answer=answers[i]
            sim_q=self.calculate_sim(question,question_weight)
            sim_a=self.calculate_sim(answer,answer_weight)
            alpha=0.5
            beta=0.5
            sim=alpha*sim_q+beta*sim_a
            qa_sim.append(sim)
        order_list=sorted(qa_sim,reverse=True)
        for i in range(int(len(order_list)/2)):
            for j in range(len(qa_sim)):
                if qa_sim[j]==order_list[i]:
                    L_qa.append(j)
                    qa_sim[j]=0
        return L_qa


    def calculate_sim(self,input_vect,weight):
        sim=np.sum(np.array(input_vect)*np.array(weight))/(np.sqrt(np.sum(np.array(input_vect)**2))*np.sqrt(np.sum(np.array(weight)**2)))
        return sim

    # def calculate_qik(self,weights,sample):
    #     for k in range(np.array(sample.shape[1])):

    def label_qa(self,question_weight,answer_weight,C_q,C_a,questions,answers,e,alpha):
        L_QA=[]


    def label_som(self,weights,C,V,e,alpha):
        """
        :param weights: 神经元i的权重向量Wi, Wi应该是k维的向量，k为特征词的数量
        :param C: 所有映射到神经元i的输入向量的结合,C为m*k，其中m为映射到第i个神经元的样本数，k为特征词的数量
        :param V: 特征词的集合
        :param e: 一个足够小的正数
        :param alpha: 权重阈值
        :return: 表示神经元i的主题词信息的标签集合Li
        """
        L=[]
        for k,tk in enumerate(V):
            tmp=0
            for xj in C:
              tmp+=np.power(weights[k]-xj[k],2)
            qik=np.sqrt(tmp)#计算神经元i中每一个特征词tk的量化误差qik
            # print(qik) #for test
            # print(weights[k])
            if qik<e and abs(weights[k])>alpha:
                L.append(tk)
        return L