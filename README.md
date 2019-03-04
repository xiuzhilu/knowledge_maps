# knowledge_maps
The structure of GHSOM neural network is improved, and 2DGHSOM algorithm is proposed.
GHSOM 副本为2DGHSOM的主程序，其中用到labelSOM和SOM，method12为typical questions-answers pairs 提取算法，算法的输入数据为331个问答对的建模后的tfidf数据，其中分别是q_weight和a_weight,q_word,a_word为tfidf建模问题和答案所用的词语
maps.xlsx为知识地图可视化的结果
labelSOM.py为神经元主题词提取算法，SOM.py为神经网络最基本的结构
merge_som.py为神经元的合并机制实现
