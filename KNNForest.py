import math
import random
from copy import deepcopy

from sklearn.model_selection import KFold
from scipy.spatial import distance
from ID3 import load_tables, select_feature, TDIDT_CUT
from ID3 import create_Features
from ID3 import TDIDT
from ID3 import IDT_basic_classifier
from ID3 import TreeNode
import numpy as np


def majority_class(examples):
    red = 0
    blue = 0
    for example in examples:
        if (example[0] == 'B'):
            blue += 1
        else:
            red += 1
    if red >= blue:
        return 'M'
    else:
        return 'B'


def evaluate(correct, assume):
    if correct == assume:
        return 0
    if correct == 'B' and assume == 'M':
        return 1
    return 10


def classify(tree, example):
    if tree.children == None:
        return tree.c
    num_of_feature = tree.feature
    diverge_val = tree.diverge_val
    if (example[num_of_feature] >= diverge_val):
        return classify(tree.children[0], example)
    else:
        return classify(tree.children[1], example)


def prune(T, V):
    if (T.children == None):
        return T
    Echild1 = []
    Echild2 = []
    for validation in V:
        if (validation[T.feature] >= T.diverge_val):
            Echild1.append(validation)
        else:
            Echild2.append(validation)
    child1 = prune(T.children[0], Echild1)
    child2 = prune(T.children[1], Echild2)
    T.children = [child1, child2]
    # T= best tree until yet
    err_prune = 0
    err_no_prune = 0
    for example in V:
        err_prune += evaluate(example[0], T.c)
        res_no_prune = classify(T, example)
        err_no_prune += evaluate(example[0], res_no_prune)
    if err_prune < err_no_prune:
        T.children = None
        T.feature = None
        T.diverge_val = None
    return T


def split_train_tabels(train_tabels,num=2):
    kf = KFold(num, True, 318965365)
    for train_index, test_index in kf.split(train_tabels):  # loop 5 times
        trainE = []
        testE = []
        for i in train_index:  # relevant E for train
            trainE.append(train_tabels[i])
        for j in test_index:  # relevant E for test
            testE.append(train_tabels[j])
        return trainE, testE

class KNN_tree:
    def __init__(self, tree, centroid):
        self.tree = tree
        self.centroid = centroid

class Forest:
    def __init__(self,E,F,number_of_trees,M,p):
        self.E = E
        self.F = F
        self.number_of_trees = number_of_trees
        self.M=M
        self.trees=[]
        self.p=p

    def create_forest(self):
        for i in range(0,self.number_of_trees):
            num_chosenE = self.p * len(self.E)
            random_E = random.sample(self.E, math.floor(num_chosenE))
            without_diag = np.delete(random_E, 0, 1)
            centroid=(np.mean(without_diag.astype(np.float), axis=0))
            self.trees.append(KNN_tree(TDIDT_CUT(random_E,self.F,majority_class(random_E),select_feature,self.M),centroid))


if __name__ == '__main__':
    train_tables = load_tables("train.csv")
    test_tables = load_tables("test.csv")

    F = create_Features(len(train_tables[0]))
    forest = Forest(train_tables,F,3,1,0.5)
    forest.create_forest()
    test_table1s = load_tables("test.csv")
