import csv
import numpy as np
from sklearn.model_selection import KFold
from math import log as logbase
from copy import deepcopy
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class TreeNode:
    def __init__(self, feature, diverge_val, children, c):
        self.feature = feature
        self.diverge_val = diverge_val
        self.children = children
        self.c = c


def TDIDT(E, F, Default, SelectFeature):
    if len(E) == 0:
        return TreeNode(None, None, None, Default)
    c = majority_class(E)
    if (is_consistent(E)):  # Consistent Node
        return TreeNode(None, None, None, c)
    f, diverge_val = SelectFeature(E, F)  # f is a number of the next feature
    # F.remove(f) - not needed
    Echild1 = []
    Echild2 = []
    for example in E:
        if (example[f] >= diverge_val):
            Echild1.append(example)
        else:
            Echild2.append(example)
    subtrees = []
    subtrees.append(TDIDT(Echild1, F, c, SelectFeature))
    subtrees.append(TDIDT(Echild2, F, c, SelectFeature))
    return TreeNode(f, diverge_val, subtrees, c)

    # early cut!


def TDIDT_CUT(E, F, Default, SelectFeature, M):
    if len(E) == 0:
        return TreeNode(None, None, None, Default)
    c = majority_class(E)
    if (is_consistent(E) or len(E) < M):  # Consistent Node
        return TreeNode(None, None, None, c)
    f, diverge_val = SelectFeature(E, F)  # f is a number of the next feature
    # F.remove(f) - not needed
    Echild1 = []
    Echild2 = []
    for example in E:
        if (example[f] >= diverge_val):
            Echild1.append(example)
        else:
            Echild2.append(example)
    subtrees = []
    subtrees.append(TDIDT_CUT(Echild1, F, c, SelectFeature, M))
    subtrees.append(TDIDT_CUT(Echild2, F, c, SelectFeature, M))
    return TreeNode(f, diverge_val, subtrees, c)


# M= list of paremeters
def K_fold(E, F, c, SelectFeature, M):
    kf = KFold(5, True, 123456789)  # TODO: change to my ID
    accuarcy_M_list = []
    for Mparmeter in M:
        sum_accuracy = 0
        for train_index, test_index in kf.split(E):  # loop 5 times
            trainE = []
            testE = []
            for i in train_index:  # relevant E for train
                trainE.append(E[i])
            for j in test_index:  # relevant E for test
                testE.append(E[j])
            tree = TDIDT_CUT(trainE, F, c, SelectFeature, Mparmeter)
            cut_classifier = IDT_basic_classifier(trainE, testE, None, tree)
            curr_accuracy = cut_classifier.predict_cut_IDT()
            sum_accuracy += curr_accuracy
        total_accuracy = sum_accuracy / 5
        accuarcy_M_list.append([Mparmeter, total_accuracy])
    return accuarcy_M_list


def load_tables(name_of_file):
    with open(name_of_file, newline='') as csvfile:
        tables = list(csv.reader(csvfile))
    features = tables[0]  ##if I need
    tables.remove(tables[0])
    max_len = len(tables[0])
    for i in range(len(tables)):
        for j in range(1, max_len):
            tables[i][j] = float(tables[i][j])


    return tables


def select_feature(examples, F):
    best_feture = 0
    max_IG = 0
    best_diverge_val = 0
    for num_of_feature in F:  # range(1, len(examples[0]) - 1):
        sorted_feature_list = []
        for example in examples:
            sorted_feature_list.append([example[num_of_feature], example[0]])
        sorted_feature_list.sort()
        # now we have sorted list
        max_IG_per_feature = 0
        best_diverge_val_per_feature = 0
        for i in range(0, len(sorted_feature_list) - 1):
            diverge_val = (sorted_feature_list[i][0] + sorted_feature_list[i + 1][0]) / 2
            curr_IG = calc_IG(sorted_feature_list, diverge_val)
            if curr_IG >= max_IG_per_feature:
                max_IG_per_feature = curr_IG
                best_diverge_val_per_feature = diverge_val
        if max_IG_per_feature >= max_IG:
            max_IG = max_IG_per_feature
            best_diverge_val = best_diverge_val_per_feature
            best_feture = num_of_feature
    return best_feture, best_diverge_val


def calc_IG(sorted_examples, diverge_val):
    group1 = []
    group2 = []
    for example in sorted_examples:
        if example[0] > diverge_val:
            group1.append(example)
        else:
            group2.append(example)
    orig_entropy = entropy(sorted_examples)
    if (len(group1) != 0):
        entropy1 = (len(group1) * entropy(group1) / len(sorted_examples))
    else:
        entropy1 = 0
    if (len(group2) != 0):
        entropy2 = (len(group2) * entropy(group2) / len(sorted_examples))
    else:
        entropy2 = 0
    return orig_entropy - (entropy1 + entropy2)


def log(num):
    if num == 0:
        return 0
    else:
        return logbase(num, 2)


def entropy(examples):
    red = 0
    blue = 0
    for example in examples:
        if (example[1] == 'B'):
            blue += 1
        else:
            red += 1
    total = len(examples)
    if (total == 0):
        print("Error in calc entropy")
        return 1
    p_blue = blue / total
    p_red = red / total
    val = -(p_blue) * log(p_blue) - ((p_red) * log(p_red))
    return val


def majority_class(examples):
    red = 0
    blue = 0
    for example in examples:
        if (example[0] == 'B'):
            blue += 1
        else:
            red += 1
    if red > blue:
        return 'M'
    else:
        return 'B'


def is_consistent(examples):
    red = 0
    blue = 0
    for example in examples:
        if (example[0] == 'B'):
            blue += 1
        else:
            red += 1
    if red == 0 or blue == 0:
        return True
    else:
        return False


def create_Features(x):
    F = []
    for i in range(1, x):
        F.append(i)
    return F


class IDT_basic_classifier:
    def __init__(self, train_tables, test_tables, TDIDT, tree=None):
        self.train_tables = train_tables
        self.test_tables = test_tables
        self.TDIDT = TDIDT
        self.tree = tree

    # get a DT and E and decide if it's R or B
    def classify(self, tree, example):
        if tree.children == None:
            return tree.c
        num_of_feature = tree.feature
        diverge_val = tree.diverge_val
        if (example[num_of_feature] >= diverge_val):
            return self.classify(tree.children[0], example)
        else:
            return self.classify(tree.children[1], example)

    def predict_IDT(self):
        c = majority_class(self.train_tables)
        F = create_Features(len(self.train_tables[0]))
        tree = self.TDIDT(self.train_tables, F, c, select_feature)
        sum = len(self.test_tables)
        counter = 0
        i = 0
        for test in self.test_tables:
            result = self.classify(tree, test)
            if (result == test[0]):
                counter += 1
        success_rate = (counter / sum)
        print(success_rate)

    def predict_cut_IDT(self):
        # c = majority_class(self.train_tables)
        # F = create_Features(len(self.train_tables[0]))
        # tree = self.TDIDT(self.train_tables, F, c, select_feature,self.M)  #cut tdidt
        sum = len(self.test_tables)
        counter = 0
        i = 0
        for test in self.test_tables:
            result = self.classify(self.tree, test)
            if (result == test[0]):
                counter += 1
        success_rate = (counter / sum)
        return success_rate


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_tables = load_tables("train.csv")
    test_tables = load_tables("test.csv")

    c = majority_class(train_tables)
    F = create_Features(len(train_tables[0]))
    basic_classifier = IDT_basic_classifier(train_tables, test_tables,TDIDT)
    basic_classifier.predict_IDT()


    x= K_fold(train_tables, F, c, select_feature, [ 120])
    print(x)


    # experiment(train_tables, F, c, select_feature, [2,40,50])
    # basic_classifier = IDT_basic_classifier(train_tables, test_tables, TDIDT)
    # basic_classifier.predict()
