# encoding=utf-8
'''
author:lyq
about:mnist classification by CART
'''
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
class mnist_classification:
    def __init__(self, aver_list, cv_list):
        self.aver_list = aver_list
        self.cv_list = cv_list
    def model(self):
        X = load_digits().data
        Y = load_digits().target
        CART = DecisionTreeClassifier()
        for c in self.cv_list:
            scores = cross_val_score(CART,X,Y,cv=c)
            sum = 0
            for s in scores:
                sum += s
            self.aver_list.append(sum/c)
            print('classification average accuracy rate: %.2f'%(sum/c))
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(self.cv_list,self.aver_list,width=0.6,facecolor='red',edgecolor='white')
        plt.title('diverse-cross-validation-classification-average-accuracy-rate')
        plt.xlabel('')
        plt.show()
        
if __name__ == '__main__':
    mc = mnist_classification(aver_list=[],cv_list=list(range(2,18)))
    mc.model()