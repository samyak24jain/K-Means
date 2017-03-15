import pandas as pd
from random import sample
from math import sqrt
from numpy import mean
import copy


def accuracy(cluster_labels, class_labels):
    county = [0,0]
    countn = [0,0]
    tp = [0, 0]
    tn = [0, 0]
    fp = [0, 0]
    fn = [0, 0]
    
    for i in range(len(df)):
        # Yes = 1, No = 0
        if cluster_labels[i] == 1 and class_labels[i] == 'Yes':
            tp[0] = tp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'No':
            tn[0] = tn[0] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'No':
            fp[0] = fp[0] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'Yes':
            fn[0] = fn[0] + 1
    
    for i in range(len(df)):
        # Yes = 0, No = 1
        if cluster_labels[i] == 0 and class_labels[i] == 'Yes':
            tp[1] = tp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'No':
            tn[1] = tn[1] + 1
        if cluster_labels[i] == 0 and class_labels[i] == 'No':
            fp[1] = fp[1] + 1
        if cluster_labels[i] == 1 and class_labels[i] == 'Yes':
            fn[1] = fn[1] + 1
    
    a0 = float((tp[0] + tn[0]))/(tp[0] + tn[0] + fn[0] + fp[0])
    a1 = float((tp[1] + tn[1]))/(tp[1] + tn[1] + fn[1] + fp[1])
    p0 = float(tp[0])/(tp[0] + fp[0])
    p1 = float(tp[1])/(tp[1] + fp[1])
    r0 = float(tp[0])/(tp[0] + fn[0])
    r1 = float(tp[1])/(tp[1] + fn[1])
    
    accuracy = [a0*100,a1*100]
    precision = [p0*100,p1*100]
    recall = [r0*100,r1*100]
    
    return accuracy, precision, recall


def initializeCenters(df, k):
    random_indices = sample(range(len(df)), k)
    centers = [list(df.iloc[idx]) for idx in random_indices]
    print("Random Indices : " + str(random_indices))
    return centers


def computeCenter(df, k, cluster_labels):
    cluster_centers = list()
    data_points = list()
    for i in range(k):
        for idx, val in enumerate(cluster_labels):
            if val == i:
                data_points.append(list(df.iloc[idx]))
        cluster_centers.append(map(mean, zip(*data_points)))
    return cluster_centers


def euclidean_distance(x, y):
    summ = 0
    for i in range(len(x)):
        term = (x[i] - y[i])**2
        summ += term
    return sqrt(summ)


def assignCluster(df, k, cluster_centers):
    cluster_assigned = list()
    for i in range(len(df)):
        distances = [euclidean_distance(list(df.iloc[i]), center) for center in cluster_centers]
        min_dist, idx = min((val, idx) for (idx, val) in enumerate(distances))
        cluster_assigned.append(idx)
    return cluster_assigned


def kmeans(df, k, class_labels):
    cluster_centers = initializeCenters(df, k)
    curr = 1
    
    while curr <= MAX_ITER:
        cluster_labels = assignCluster(df, k, cluster_centers)
    #     print (cluster_labels)
        prev_centers = copy.deepcopy(cluster_centers)
    #     print("Previous Cluster Centers: \n")
        cluster_centers = computeCenter(df, k, cluster_labels)
        curr += 1
#         print("Cluster 0: " + str(euclidean_distance(prev_centers[0], cluster_centers[0])))
#         print("Cluster 1: " + str(euclidean_distance(prev_centers[1], cluster_centers[1])))
    
    return cluster_labels, cluster_centers


k = 2
MAX_ITER = 100 
df_full = pd.read_csv('SPECTF_New.csv')
columns = list(df_full.columns)
features = columns[:len(columns)-1]
class_labels = list(df_full[columns[-1]])
df = df_full[features]


labels, centers = kmeans(df, k, class_labels)
a,p,r = accuracy(labels, class_labels)

# cluster0 = 0
# cluster1 = 0
# for i in labels:
#     if i == 0:
#         cluster0 += 1
#     else:
#         cluster1 += 1
# print("Number of data points in Cluster 0: " + str(cluster0))
# print("Number of data points in Cluster 1: " + str(cluster1))


print("Accuracy = " + str(a))
print("Precision = " + str(p))
print("Recall = " + str(r))
