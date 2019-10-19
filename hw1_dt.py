import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        self.pruned=False
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    
    def split(self):
        #print(self.features)
        if(self.features is None or self.labels is None):
           
            self.splittable = False
            return
        m=len(self.features)
        n = len(self.features[0])
        if(n==0 or m==0):
            
            self.splittable=False
            return
        # FOR ALL COLUMNS
        # for ith each column
        maxIG = -np.inf
        UniqueFeatures = -1
        Labels = self.labels
        total_labels = len(Labels)
       
        NUniqueLabels = np.unique(np.array(Labels))
        MinUniqueSubLabels = -1
        entropy = 0
   
        for i in NUniqueLabels:
            count_i = Labels.count(i)
            if (count_i != 0 and total_labels != 0):
                entropy = entropy - (count_i / total_labels) * np.log2(count_i / total_labels)
                
       
        
        for i in range(n):

            Attr = np.array(self.features)[:, i]
            # finding unique values of ith coulumn
            #print(Attr)
            #break
            NUniqueSubAttr = np.unique(np.array(Attr))
            if(len(NUniqueSubAttr)==1):
                continue
                
            branches = []
            # for each unique value of ith column
            for j in sorted(NUniqueSubAttr):
                # Counting indexof features where this column is present in features

                Index_i = []
                subbranch = []
                count=0
                for row in self.features:
                    if (row[i] == j and row[i]is not None):
                        Index_i.append(count)
                    count += 1

                # Index_i.append((np.array(self.features[i])).index(j))

                # take labels for each NUniueq attributes
                Labels_i = []

                for k in Index_i:
                    Labels_i.append(self.labels[k])

                for k in sorted(NUniqueLabels):
                    subbranch.append(Labels_i.count(k))

                branches.append(subbranch)

            # calculating entropy for given coulmn
            # in whole feature we are calculating no of yes and no of nos
            #self.feature_uniq_split = NUniqueSubAttr
                     
            IG = Util.Information_Gain(entropy, branches)
           
                
            
            
            if (IG > maxIG):
                maxIG = IG
                self.dim_split = i
                self.feature_uniq_split = NUniqueSubAttr
                #MinUniqueSubLabels = self.feature_uniq_split
            elif (IG == maxIG):
                if (len(self.feature_uniq_split) < len(NUniqueSubAttr)):
                    maxIG = IG
                    self.dim_split = i
                    self.feature_uniq_split = NUniqueSubAttr
                    #MinUniqueSubLabels = self.feature_uniq_split
                elif (len(self.feature_uniq_split) == len(NUniqueSubAttr)):
                    if (self.dim_split > i):
                        maxIG = IG
                        self.dim_split = i
                        self.feature_uniq_split = NUniqueSubAttr
                        #MinUniqueSubLabels = self.feature_uniq_split
        
            
        if (self.feature_uniq_split is not None and len(self.feature_uniq_split)>1 and self.labels is not None and self.dim_split is not None):
            count = 0
            for i in self.feature_uniq_split:
                NewFeature = []
                NewLabels = []
                c=0
                for j in self.features:
                    if (j[self.dim_split] == i):
                        NewFeature.append(j)
                        NewLabels.append(self.labels[c])
                    c+=1

                NoOfUniqueLabels = len(np.unique(NewLabels))
                NewFeature = np.delete(np.array(NewFeature), self.dim_split, 1)
                NewC=TreeNode(NewFeature, NewLabels, NoOfUniqueLabels)
                self.children.append(NewC)
                #ChildrenDict[NewC] = NoOfUniqueLabels
                
                count += 1
            for t in self.children:
                if (t.splittable == True):
                    t.split()      
        #listofTuples = sorted(ChildrenDict.items(), key=lambda x: x[1])
        #for item in listofTuples:
        #    self.children.append(item[0])
        else:
            
            return
        
            
        
        
        
        return
        #raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if len(feature) > 0:
            
            if self.splittable and self.feature_uniq_split is not None:
                for x in range(len(self.feature_uniq_split)):
                    if feature[self.dim_split] == self.feature_uniq_split[x]:
                        child = x
                        NewFeature=[]
                        for i in range(len(feature)):
                            if(i!=self.dim_split):
                                NewFeature.append(feature[i])
                        if(self.children[child].pruned==False):
                            return self.children[child].predict(NewFeature)
                        else:
                            return self.cls_max
                return self.cls_max
            else:
                return self.cls_max
        else:
            return self.cls_max
        #raise NotImplementedError