"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Perceptron Implementation ***
Paper: Bifet, Albert, et al. "Fast perceptron decision tree learning from evolving data streams."
Published in: Advances in knowledge discovery and data mining (2010): 299-310.
URL: http://www.cs.waikato.ac.nz/~eibe/pubs/Perceptron.pdf
"""

import math
import operator
import random
# import miscMethods
from collections import OrderedDict
import numpy as np
from classifier.classifier import SuperClassifier
from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic
from .miscMethods import *
from sklearn import preprocessing

class OLIFL(SuperClassifier):
    """This is the implementation of OLIFL for learning from data streams."""

    LEARNER_NAME = TornadoDic.OLIFL
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    __BIAS_ATTRIBUTE = Attribute()
    __BIAS_ATTRIBUTE.set_name("bias")
    __BIAS_ATTRIBUTE.set_type(TornadoDic.NUMERIC_ATTRIBUTE)
    __BIAS_ATTRIBUTE.set_possible_values(1)

    def __init__(self, labels, attributes, alpha, FR, learning_rate=1):
        super().__init__(labels, attributes)

        attributes.append(self.__BIAS_ATTRIBUTE)
        self.WEIGHTS = OrderedDict()
        self.fremove = FR
        self.remove = 0
        self.x_attributes = OrderedDict()
        self.features_left = 1
        self.remainNum = 10
        self.__initialize_weights()
        self.LEARNING_RATE = learning_rate
        self.C = 0.01
        self.T = 0.001
        self.gamma  = 0.00001
        self.Is_Drift = False
        self.alpha = alpha
        # print(self.alpha)
        # self.features_left = 10

    def rFeatures(self, fremove):#"variable"
        for a in self.ATTRIBUTES:
            if np.random.random()>(1-fremove):
                self.x_attributes[a.NAME] = 0
            else:
                self.x_attributes[a.NAME] = 1
        self.x_attributes["bias"] = 1


    def rDataTrapezoidal(self,features_left): #"Trapezoidal"
        key=0
        for a in self.ATTRIBUTES:
            if key > features_left:
                self.x_attributes[a.NAME] = 0
            else:
                self.x_attributes[a.NAME] = 1
            key+=1
        self.x_attributes["bias"] = 1

    def rDataEvolvable(self,features,flag): #"evolvable"
        key=0
        if flag == 1:
            for a in self.ATTRIBUTES:
                if key > len(self.ATTRIBUTES)-features:
                    self.x_attributes[a.NAME] = 0
                else:
                    self.x_attributes[a.NAME] = 1
                key+=1
        elif flag == 2:
            for a in self.ATTRIBUTES:
                if key > len(self.ATTRIBUTES)-features:
                    self.x_attributes[a.NAME] = 0
                else:
                    self.x_attributes[a.NAME] = 1
                key+=1
        self.x_attributes["bias"] = 1

    def __initialize_weights(self):
        self.rFeatures(self.fremove)#"variable"
        # self.rDataTrapezoidal(self.features_left)#"Trapezoidal"
        for c in self.CLASSES:
            self.WEIGHTS[c] = OrderedDict()
            for a in self.ATTRIBUTES:
                if self.x_attributes[a.NAME] == 1:#只初始化，特征存在的相应权重
                    self.WEIGHTS[c][a.NAME] = 0.2 * random.random() - 0.1
        self.stability = OrderedDict()
        self.A = dict()
        self.A_= dict()
        self.keyCount = dict() #权重策略，初始化为1
        self.count = dict() #计数，统计出现多少次
        self.R = 0.00001

    def parameter_set(self, x, loss):
        # inner_product = np.sum([X[k]*self.keyCount[k] * X[k]*self.keyCount[k] for k in X.keys()])
        inner_product = sum(np.multiply(x, x))
        return np.minimum(self.C, 2*loss / inner_product)


    def train_semi(self, instance, erroNum):
        self.rFeatures(self.fremove)#"variable"
        # self.rDataTrapezoidal(self.features_left)#"Trapezoidal"
        # self.rDataEvolvable(self,)
        x = instance[0:len(instance) - 1]
        x.append(1)
        # x = preprocessing.scale(x).tolist()
        y_real = instance[len(instance) - 1]
        predictions = OrderedDict()
        # tao = self.parameter_set(x)
        for c in self.CLASSES:
            predictions[c] = self.predict(x, c)

        for c in self.CLASSES:# 理论可行
            #半监督计算
            #当不是此类的分类器时不计算，减少时间
            actual = 1 if c == y_real else 0
            if np.random.random() > (1-self.remove): # randomly set unlabeled instance,
                theta = self.upper_bound(x,self.WEIGHTS[c])#標簽越少，誤差數越少，限制越不精確？
                theta = erroNum/theta
                if theta < self.T:#置信度过低，抛弃实例
                    self.keyCount = self.e_KeyCount
                    self.stability = self.e_stability
                    self.weights = self.e_weights
                    continue
                else:
                    # print("select!!!")
                    actual = 1
                    predictions[c] = theta * predictions[c]

        # loss = (np.maximum(0, (1 - predictions[c] * int(y_real))))
            # delta = self.parameter_set(x, loss)
            delta = (actual - predictions[c]) * predictions[c] * (1 - predictions[c])

            # delta = int(y_real)-predictions[c]* predictions[c] * (1 - predictions[c])
            for i in range(0, len(instance)):
                if self.x_attributes[self.ATTRIBUTES[i].NAME] == 1:
                    self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] += self.LEARNING_RATE * delta * x[i] * self.keyCount[self.ATTRIBUTES[i].NAME]
                    # self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] += self.LEARNING_RATE * delta * x[i]
                    # self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] += self.LEARNING_RATE * delta * x[i] * self.stability[self.ATTRIBUTES[i].NAME]
        self._IS_READY = True

    def predict(self, x, c):
        s = 0
        # x = instance[0:len(instance) - 1]
        #列表转字典，方便操作，
        arr = list()
        for a in self.ATTRIBUTES:
            arr.append(a.NAME)
        X = OrderedDict(zip(arr,x))
        for key in self.x_attributes:
            if self.x_attributes[key] == 0 :
                del X[key]
        X = self.expand_space(X)
        self.update_stability(X)
        self.upKeyCount(X)
        # print(self.keyCount)
        for i in range(0, len(x)):
            if self.x_attributes[self.ATTRIBUTES[i].NAME] == 1:
                s += self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] * x[i] * self.keyCount[self.ATTRIBUTES[i].NAME]
                # s += self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] * x[i] * self.stability[self.ATTRIBUTES[i].NAME]

                # s += self.WEIGHTS[c][self.ATTRIBUTES[i].NAME] * x[i]
        p = 1 / (1 + math.exp(-s))#sigmod 激活函数
        return p


    # def test(self, instance):

    def test(self, instance, features_left):
        self.features_left = features_left
        if self._IS_READY:
            x = instance[0:len(instance) - 1]
            y = instance[len(instance) - 1]
            x.append(1)
            predictions = OrderedDict()
            for c in list(self.CLASSES):
                predictions[c] = self.predict(x, c)
            y_predicted = max(predictions.items(), key=operator.itemgetter(1))[0]
            self.update_confusion_matrix(y, y_predicted)
            return y_predicted
        else:
            print("Please train a OLIFL classifier first!")
            exit()

    def reset(self):
        self.Is_Drift = True
        super()._reset_stats()
        # self.WEIGHTS = OrderedDict()
        # self.__initialize_weights()



    def expand_space(self, X):#补全特征空间,并且将实例和分类器扩充到全局特征空间
        self.n_keys = dict() #定义新特征
        self.e_keys = dict()
        # self.s_keys = dict()

        self.e_weights = self.WEIGHTS#每个类都有一个weight，但是feature space是相同的
        for key in findDifferentKeys(X, self.WEIGHTS[list(self.CLASSES)[0]]):
            for c in list(self.CLASSES):
                self.WEIGHTS[c][key] = 0
            self.n_keys[key] = 1
        for key in findDifferentKeys(self.WEIGHTS[list(self.CLASSES)[0]],X):
            X[key] = 0
            self.e_keys[key] = 1
        # for key in findCommonKeys(X,self.WEIGHTS[list(self.CLASSES)[0]]):
        #     self.u_count[key] += 1
        #     self.s_keys=+1
        return X

    def update_stability(self, X):# feature space 扩充完毕，直接输入扩充过后的实例

        self.e_stability = self.stability
        getCount = self.count.get
        getA = self.A.get
        for key in X.keys():
            if key not in self.count.keys():
                self.count[key] = 1
                self.keyCount[key]=1#最开始出现，权重应该大，信息量大
                self.A_[key] = X[key]
                self.A[key] = X[key]
                self.stability[key]=0.0000001#最开始出现时，初始权重
            else:
                self.count[key] +=1
                self.A_[key] = getA(key)
                self.A[key] = getA(key) + (X[key] - getA(key)) / getCount(key)
                self.stability[key]=(getCount(key)-1)/getCount(key)**2*(X[key]-getA(key))**2+(getCount(key)-1)/getCount(key)*self.stability[key]

    def upKeyCount(self,X):  # 方差比例越大信息量越大,出现为1，不出现为零，统计离散度
        sum1 = 0
        self.e_KeyCount=self.keyCount
        if self.Is_Drift:# 若发生漂移，降低旧特征权重
            self.upEKeys()
        getStability = self.stability.get

        for key in X.keys():
            sum1 += getStability(key)
        for key in X.keys():
            self.keyCount[key] = getStability(key)/ sum1

    def upEKeys(self):
        for key in self.e_keys:
            # print(getStability(key))
            # self.stability[key] = 0.000001 * self.stability[key]
            self.stability[key] = self.alpha * self.stability[key]

    def upper_bound(self,x,weights):
        # x=list(x)
        # w=list(self.weights)
        x_norm = math.sqrt(np.dot(x,x))
        w_norm = math.sqrt(dotDict(weights,weights))
        # gamma  = self.min_gamma
        if x_norm > self.R:
            self.R = x_norm
        theta = self.R * w_norm / self.gamma
        if theta == 0:
            theta=0.1
        return theta