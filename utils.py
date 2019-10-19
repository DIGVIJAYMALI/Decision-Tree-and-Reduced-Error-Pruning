import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    sum=np.sum(branches)
    entropy=0
    for i in branches:
        s1=np.sum(i)
        x=EntropyFun(i)
        entropy=entropy+(s1/sum)*x



    InfGain=S-entropy
    #print(InfGain)
    return InfGain
    raise NotImplementedError

def EntropyFun(i):
    entropy=0
    #print("!!!!!!")
    #print(i)
    sum=np.sum(i)
    for j in i:
        if(sum!=0 and j!=0):
            entropy=entropy-((j/sum)*np.log2(j/sum))
    return entropy


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    if(decisionTree and len(X_test)!=0 and X_test is not None):
        root=decisionTree.root_node
        if root is not None:
            REP(root,X_test,y_test,root)
        return
    return
    #printPruned(root)

def REP(node,X_test,y_test,root):

    if(node.splittable==True):
        
        for child in reversed(node.children):
         
            REP(child, X_test,y_test,root)

        # start pruning when recursion stops
        predictions1 = CalPred(root, X_test)
        correctPred = 0
        n = len(y_test)
        for i in range(n):
            if (predictions1[i] == y_test[i]):
                correctPred += 1
        if (n != 0):
            Accuracy1 = correctPred / n
        node.pruned = True
        node.splittable=False

        predictions2 = CalPred(root, X_test)

        correctPred = 0
        n = len(y_test)
        for i in range(n):
            if (predictions2[i] == y_test[i]):
                correctPred += 1
        if (n != 0):
            Accuracy2 = correctPred / n

        if Accuracy1 > Accuracy2:
            node.pruned = False
            node.splittable=True
        else:
            node.pruned=True
            node.splittable=False

        return
    return

def CalPred(root,X_test):
    predictions = []
    for row in X_test:
        predictions.append(root.predict(row))
    return predictions

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
