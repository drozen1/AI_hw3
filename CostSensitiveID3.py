from copy import deepcopy

from sklearn.model_selection import KFold

from ID3 import load_tables, select_feature, TDIDT_CUT
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


def split_train_tabels(train_tabels, num=2):
    kf = KFold(num, True, 318965365)  # TODO: change to my ID
    for train_index, test_index in kf.split(train_tabels):  # loop 5 times
        trainE = []
        testE = []
        for i in train_index:  # relevant E for train
            trainE.append(train_tabels[i])
        for j in test_index:  # relevant E for test
            testE.append(train_tabels[j])
        return trainE, testE


def Kf_train_tabels(train_tabels, num=2, M=1, ):
    best_lost = 2
    best_tree = None
    kf = KFold(num, True, 318965365)  # TODO: change to my ID
    for train_index, test_index in kf.split(train_tabels):  # loop 5 times
        trainE = []
        testE = []
        for i in train_index:  # relevant E for train
            trainE.append(train_tabels[i])
        for j in test_index:  # relevant E for test
            testE.append(train_tabels[j])
        c = majority_class(trainE)
        F = create_Features(len(trainE[0]))
        T = TDIDT_CUT(trainE, F, c, select_feature, M)
        TT = prune(T, testE)
        basic_classifier = IDT_basic_classifier(None, testE, None, TT)
        curr_loss = basic_classifier.predict_IDT_loss(False)
        # print(curr_loss)
        if (curr_loss < best_lost):
            best_lost = curr_loss
            best_tree = TT
    return best_tree #,best_lost


if __name__ == '__main__':
    train_tables = load_tables("train.csv")
    test_tables = load_tables("test.csv")
    # date3 = load_tables("data3.csv")
    # date = load_tables("data.csv")
    T = Kf_train_tabels(train_tables, 4)

    basic_classifier = IDT_basic_classifier(None, test_tables, None, T)
    x = basic_classifier.predict_IDT_loss(False)
    print(x)
    # basic_classifier = IDT_basic_classifier(None, date3, None, T)
    # x = basic_classifier.predict_IDT_loss(False)
    # print(x)
    #0.038461538461538464 without prune m=3 k=4
    # xtest=0
    # Mtest, Mdata, Mpredicted =0,0,0
    # xdata3=0
    # xpredicted = 2
    # Ktest, Kdata, Kpredicted = 0, 0,0
    # for M in range(1,6):
    #     for k in range(2, 6):
    #         print("M is: ",M, "K is: ",k)
    #         T,curr_loss = Kf_train_tabels(train_tables, k, M)
    #         basic_classifier = IDT_basic_classifier(None, test_tables, None, T)
    #         print("predicted loss is: ",curr_loss)
    #         if xpredicted>curr_loss:
    #             Kpredicted=k
    #             Mpredicted=M
    #             xpredicted=curr_loss
    #         x = basic_classifier.predict_IDT_loss(False)
    #        # print("loss on test is: ",x)
    #         if xtest>x:
    #             Ktest=k
    #             Mtest=M
    #             xtest=x
    #         basic_classifier = IDT_basic_classifier(None, date3, None, T)
    #         x = basic_classifier.predict_IDT_loss(False)
    #    #     print("loss on data3 is: ",x)  # 0.038461538461538464 without prune m=3 k=4
    #         if xdata3>x:
    #             Kdata=k
    #             Mdata=M
    #             xdata3=x
    #
    # print("for test best vals are: M:", Mtest, "K: ", Ktest, "loss is: ", xtest)
    # print("for date3 best vals are: M:", Mdata, "K: ", Kdata, "loss is: ", xdata3)
    # print("for predicted best vals are: M:", Mpredicted, "K: ", Kpredicted, "loss is: ",xpredicted )
    # best_tree = prune(T, V_test)
    # basic_classifier1 = IDT_basic_classifier(None, test_tables, None, best_tree)
    # loss = basic_classifier1.predict_IDT_loss(False)
    # print(loss)

    # train_tables = load_tables("train.csv")
    # test_tables = load_tables("test.csv")
    # V_train, V_test = split_train_tabels(train_tables,4)
    # c = majority_class(V_train)
    # F = create_Features(len(V_train[0]))
    # T = TDIDT_CUT(V_train, F, c, select_feature,3)
    #
    # basic_classifier = IDT_basic_classifier(None, test_tables, None, T)
    # x = basic_classifier.predict_IDT_loss(False)
    # #print(x)
    # best_tree = prune(T, V_test)
    # basic_classifier1 = IDT_basic_classifier(None, test_tables, None, best_tree)
    # loss = basic_classifier1.predict_IDT_loss(False)
    # print(loss)

    """
      ניסיון לכוונן ערכים:
    """

    # minloss=1
    # minloss_index=-1
    # min_lossM=-1
    # F = create_Features(len(train_tables[0]))
    # for M in range (1,10):
    #     print(M, " mParemeter")
    #     for i in range (2,5):
    #         print(i)
    #         V_train, V_test = split_train_tabels(train_tables, i)
    #         c = majority_class(V_train)
    #         cut_tree = TDIDT_CUT(V_train, F, c, select_feature, M)
    #         basic_classifier = IDT_basic_classifier(None, test_tables, None, cut_tree)
    #         x = basic_classifier.predict_IDT_loss(False)
    #         print(x)
    #         best_tree = prune(cut_tree, V_test)
    #         basic_classifier1 = IDT_basic_classifier(None, test_tables, None, best_tree)
    #         x = basic_classifier1.predict_IDT_loss(False)
    #         print(x)
    #         if(x<minloss):
    #             minloss=x
    #             minloss_index=i
    #             min_lossM=M
    #
    # print(minloss_index,minloss,min_lossM)

    # M=2 0.012389380530973451 k=2
    # M=12 0.001769911504424779 k=3
    # K=4 0.004424778761061947 no early cut
