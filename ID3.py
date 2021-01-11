import csv
from math import log as logbase

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def load_tables(name_of_file):
    with open(name_of_file, newline='') as csvfile:
        tables = list(csv.reader(csvfile))
    features=tables[0] ##if I need
    tables.remove(tables[0])
    for i in range(len(tables)):
        for j in range(1,len(tables[0])):
            if j==len(tables[0])-1:
                tables[i].remove(tables[i][j])
            else:
                tables[i][j] = float(tables[i][j])

    return tables

def select_feature(examples):
    best_feture=0
    max_IG=0
    best_diverge_val=0
    for num_of_feature in range(1,len(examples[0])-1):
        sorted_feature_list = []
        for example in examples:
            sorted_feature_list.append([example[num_of_feature],example[0]])
        sorted_feature_list.sort()
        #now we have sorted list
        max_IG_per_feature=0
        best_diverge_val_per_feature = 0
        for i in range(0,len(sorted_feature_list)-1):
            diverge_val=(sorted_feature_list[i][0]+sorted_feature_list[i+1][0])/2
            curr_IG=calc_IG(sorted_feature_list,diverge_val)
            if curr_IG>=max_IG_per_feature:
                max_IG_per_feature=curr_IG
                best_diverge_val_per_feature=diverge_val
        if max_IG_per_feature>=max_IG:
            max_IG=max_IG_per_feature
            best_diverge_val=best_diverge_val_per_feature
            best_feture=num_of_feature
    return best_feture,best_diverge_val

def calc_IG(sorted_examples, diverge_val):
    group1=[]
    group2=[]
    for example in sorted_examples:
        if example[0]>diverge_val:
            group1.append(example)
        else:
            group2.append(example)
    orig_entropy=entropy(sorted_examples)
    if(len(group1)!=0):
        entropy1=(len(group1)*entropy(group1)/len(sorted_examples))
    else:
        entropy1=0
    if (len(group2) != 0):
        entropy2=(len(group2)*entropy(group2)/len(sorted_examples))
    else:
        entropy2=0
    return orig_entropy-(entropy1+entropy2)

def log(num):
    if num == 0:
        return 0
    else:
        return logbase(num,2)


def entropy(examples):
    red=0
    blue=0
    for example in examples:
        if(example[1]=='B'):
            blue+=1
        else:
            red+=1
    total=len(examples)
    if(total==0):
        print("Error in calc entropy" )
        return 1
    p_blue=blue/total
    p_red = red / total
    val= -(p_blue)*log(p_blue)-((p_red)*log(p_red))
    return val

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_tables=load_tables("train.csv")
    select_feature(train_tables)
    test_tables=load_tables("test.csv")

