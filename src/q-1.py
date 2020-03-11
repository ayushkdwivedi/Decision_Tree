'''
Command to run the script:  python q-1.py --data_type <0/1/2> --measure_type <0/1/2>

data_type 0: Numerical Data     1: Categorical Data measure_type    2: Complete Data measure_type

measure_type 0: Entropy     1: Gini     2: Misclassification
'''

from __future__ import division
import sys
import numpy as np
import pandas as pd
import math
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


classes = ["0", "1"]

class decision_node:
    def __init__(self,col=-1,value=None, results=None, true_branch=None, false_branch=None): # Init - Automatically runs object of class
        self.col = col                                  # Self - Making values preassigned as and when the class is called
        self.value = value
        self.results = results
        self.true_branch = true_branch
        self.false_branch = false_branch
        
    def log2(self,p):
        if p == 0:
            return 0.0
        return math.log(p, 2)
    
    def split_set(self, data, column, value):
        set_true = []
        set_false = []
        for row in data:
            if isinstance(value, int) or isinstance(value, float):
                if(row[column] >= value):
                    set_true.append(row)
                else:
                    set_false.append(row)
            else:
                if(row[column] == value):
                    set_true.append(row)
                else:
                    set_false.append(row)
        return np.array(set_true), np.array(set_false)

    def count_class(self, data):
        results = {}
        for row in data:
            r = row[-1]
            if r not in results:
                results[r]=0
            results[r]+=1
        return results

    def information_Measure(self, data, measure): 
        counts = self.count_class(data)
        total = len(data)
        ent, gi, mc, res = 0.0, 1.0, 1.0, 0.0
        pp = []
        for c in counts:
            p = float(counts[c]/total)
            if measure == 'Entropy':
                ent = float(ent-(p * self.log2(p)))
                res = ent
            elif measure == 'Gini':
                gi = float(gi - p**2)
                res = gi
            else:
                pp.append(p)
        if measure == 'Miss Classification':
            if pp == []:
                res = mc
            else:
                mc = float(mc - max(pp))
                res = mc
        return float(res)

    def create_tree(self, data, measure):
        if(len(data))==0:
            return decision_node()
        max_inf_gain = 0.0
        best_criteria = None
        best_sets = None
        similarity = self.information_Measure(data, measure)   
        
        for col in range(len(data[0])-1):
            col_values = {}
            for row in data:
                col_values[row[col]]=1
            for value in col_values.keys():
                (set_true, set_false) = self.split_set(data, col, value)
#                print(len(set_true),len(data))
                p=float(len(set_true)/len(data))
#                print("p is",p)
                inf_gain = float(similarity-p*self.information_Measure(set_true, measure)-(1-p)*self.information_Measure(set_false, measure))
#                print("ig is",inf_gain)
                if(inf_gain>max_inf_gain) and len(set_true)>0 and len(set_false)>0:
                    max_inf_gain = inf_gain
                    best_criteria = (col, value)
                    best_sets = (set_true,set_false)


        if(max_inf_gain>0):
            true_branch = self.create_tree(best_sets[0], measure)
            false_branch = self.create_tree(best_sets[1], measure)
            return decision_node(col=best_criteria[0],value = best_criteria[1],true_branch=true_branch, false_branch=false_branch)
        else:
#            print(max_inf_gain)
            return decision_node(results=self.count_class(data))
    
    def classify(self, observation, tree):
        if(tree.results != None):
            return tree.results
        else:
            v = observation[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if(v > tree.value):
                    branch = tree.true_branch
                else:
                    branch = tree.false_branch
            else:
                if v == tree.value:
                    branch = tree.true_branch
                else:
                    branch = tree.false_branch
        return self.classify(observation,branch)

def validation(tree, x_val, y_val):

    y_pred = np.zeros(x_val.shape[0])
    for i, row in enumerate(x_val[1:]):
        # pdb.set_trace()
        class_pred = tree.classify(row, tree)
        for k in class_pred.keys():
            y_pred[i] = k
    # pdb.set_trace()
    accuracy = np.sum(y_val == y_pred) / x_val.shape[0] #shape[0] = no. of rows
    recall = recall_score(y_val, y_pred, average = 'weighted')
    # print('recall', recall)
    
    precision = precision_score(y_val, y_pred, average = 'weighted', labels=np.unique(y_pred))
    f1Score = (2*recall*precision)/(recall + precision)
    return accuracy , recall, precision, f1Score

def predict(tree, x_test):

    y_pred = np.zeros(x_test.shape[0])
    for i, row in enumerate(x_test[1:]):
        # pdb.set_trace()
        class_pred = tree.classify(row, tree)
        for k in class_pred.keys():
            y_pred[i] = k
    return y_pred

def sklearn_tree(x_train, y_train, x_val, y_val):

    x_train = pd.get_dummies(x_train)
    x_val = pd.get_dummies(x_val)
    dtree = DecisionTreeClassifier(criterion = 'entropy').fit(x_train, y_train)
    print("SK Training: %s" % dtree.score(x_train, y_train))
    print("SK Validation: %s" % dtree.score(x_val, y_val))

def get_data(train, test, data_type):

    train_data = pd.read_csv(train)
    train_data = train_data.iloc[30:50, :] #To select part of data for faster running
    test_data = pd.read_csv(test)

    y = train_data.iloc[:, 6]
    train_data = train_data.drop('left', axis = 1) # Removing label from data frame
    # print('train_data',train_data)
    if data_type == 'Numerical':
        x = train_data.iloc[:, :6] # First 5 coloums denotes Numerical data
        x_test = test_data.iloc[:, :6]
    elif data_type == 'Categorical':
        x = train_data.iloc[:, 6:] # Beyond 5th coloum denotes Categorical data
        x_test = test_data.iloc[:, 6:]
    else:
        x = train_data
        x_test = test_data

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)    # Data is splitted into train and validation data

    return x_train, y_train, x_val, y_val, x_test

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--sklearn', action = 'store_true', help = 'For sklearn results')
    parser.add_argument('--data_type', required = True, help = 'Enter data type: 0-Numerical, 1-Categorical, 2-Both')
    parser.add_argument('--measure_type', required = True, help = 'Enter data type: 0-Entropy, 1-Gini, 2-Miss Classification')
    args = parser.parse_args()

    data_type = ['Numerical', 'Categorical', 'Both (Numerical and Categorical)']
    measure_type = ['Entropy', 'Gini', 'Miss Classification']
    
    train = './train.csv'
    test = './sample_test.csv'

    x_train, y_train, x_val, y_val, x_test = get_data(train, test, data_type[int(args.data_type)])
    # print('type: ',type(x_train))
    
    if args.sklearn:
        sklearn_tree(x_train, y_train, x_val, y_val)
    else:
        train_data = np.array(pd.concat([x_train, y_train], axis = 1))
        x_val = np.array(x_val)    
        x_test = np.array(x_test)
        
        dn = decision_node()
        tree = dn.create_tree(train_data, measure_type[int(args.measure_type)])

        print("Data Type: ", data_type[int(args.data_type)])
        print("Information Measure: ", measure_type[int(args.measure_type)])
        print("Training Accuracy: %s, Recall: %s, Precision: %s, f1Score: %s" % validation(tree, x_train, y_train))
        print("Validation Accuracy: %s, Recall: %s, Precision: %s, f1Score: %s" % validation(tree, x_val, y_val))
        print("Predicted Labels: %s" % predict(tree, x_test))

    df = pd.DataFrame(train_data)

    # for row in train_data:
    #     print(df.loc[:,'satisfaction_level'])
    #     plt.figure()
    #     plt.scatter(df['satisfaction_level'], df['last_evaluation'])
    # # plt.show()