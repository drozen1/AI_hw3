from copy import deepcopy

from sklearn.model_selection import KFold

from ID3 import load_tables, select_feature
from ID3 import create_Features
from ID3 import TDIDT
from ID3 import IDT_basic_classifier
from ID3 import TreeNode


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
    classifier_Tree = IDT_basic_classifier(None, V, None, T)
    err_no_prune = classifier_Tree.predict_IDT_loss(False)
    CutT = deepcopy(T)
    CutT.children = []
    CutT.feature = None
    CutT.diverge_val = None
    classifier_Cut_Tree = IDT_basic_classifier(None, V, None, CutT)
    err_prune = classifier_Tree.predict_IDT_loss(False)
    if err_prune < err_no_prune:
        return CutT
    return T


def split_train_tabels(train_tabels):
    kf = KFold(2, True, 318965365)  # TODO: change to my ID
    for train_index, test_index in kf.split(train_tabels):  # loop 5 times
        trainE = []
        testE = []
        for i in train_index:  # relevant E for train
            trainE.append(train_tabels[i])
        for j in test_index:  # relevant E for test
            testE.append(train_tabels[j])
        return trainE, testE


if __name__ == '__main__':
    train_tables = load_tables("train.csv")
    test_tables = load_tables("test.csv")
    c = majority_class(train_tables)
    F = create_Features(len(train_tables[0]))

    V_train, V_test = split_train_tabels(train_tables)
    T = TDIDT(V_train, F, c, select_feature)

    best_tree = prune(T, V_test )

    basic_classifier = IDT_basic_classifier(train_tables, test_tables, None, best_tree)
    x = basic_classifier.predict_IDT_loss(False)
    print(x)
