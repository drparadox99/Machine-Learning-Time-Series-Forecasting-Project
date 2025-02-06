#Tslearn libraries
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
#Keras libraries
from keras.models import Model,Sequential
from keras.layers import Dense,LSTM,GRU,SimpleRNN,Flatten,TimeDistributed,Input,Dense,Flatten,RepeatVector
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.optimizers import Adam
#Numpy libraries 
import numpy as np
from numpy import array,hstack,vstack,array
#Numpy libraries
import pandas as pd
#Sklearn libraries
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from time import perf_counter
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import csv

np.random.seed(0)

def importData(filePath):
    dataset = pd.read_csv(filePath) #
    return dataset


def exstract_dataset(data,col_name="date"):
    # dataset: (samples,series)
    dataset = data.drop(col_name, axis=1)
    dataset = dataset.to_numpy()
    return dataset #dataset: (samples,series)

def normalizeDataset(dataset):
    # dataset: (series,samples)
    norm_seqs = []
    lst_scalers = []

    for series in dataset:
        scaler = MinMaxScaler()
        #scaler = preprocessing.StandardScaler()
        # reshape input into numpy 2 D
        reshaped_2d_series = series.reshape(-1, 1)
        scaler.fit(reshaped_2d_series)
        norm_series = scaler.transform(reshaped_2d_series)
        # norm_pseudoSequence = norm_pseudoSequence.squeeze().float()    #reshape back to 1 D
        norm_series = norm_series.reshape(-1, )  # reshape back to 1 D

        norm_seqs.append(norm_series)
        lst_scalers.append(scaler)
    # return: (samples,series)
    return np.asarray(norm_seqs).T, lst_scalers


def determineKmeansClusters(series_scaled,numberOfSeries):
    #ELBOW METHOD: DETERMINING THE BEST NUBMER OF CLUSTERS
    start = perf_counter()
    elbow_data = []
    #determnine the appropriate number of clusters
    print("Determining the appropriate number of clusters ... ")
    for n_clusters in range (1,numberOfSeries,1):
        #print(n_clusters)
        km = TimeSeriesKMeans(n_clusters=n_clusters,max_iter=1000, metric="euclidean", verbose=False, random_state=1994,n_jobs=-1)
        y_pred = km.fit_predict(series_scaled)
        elbow_data.append((n_clusters, km.inertia_))
    end = perf_counter()
    print("Training time :  " + str(end-start))
    #Plot elbow curve
    pd.DataFrame(elbow_data,columns=['clusters', 'distance']).plot(x='clusters',y='distance')
    print("Elbow data :  \n" + str(elbow_data) )

    x = pd.DataFrame(elbow_data,columns=['clusters', 'distance'])["clusters"]
    y = pd.DataFrame(elbow_data,columns=['clusters', 'distance'])["distance"]

    # Plotting the Graph
    plt.plot(x, y)
    plt.title("Elbow plot")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS") #within cluster sum of square
    plt.savefig("National_illness_"+ str(end-start) +"_"+ ".png")

    #plt.show()


def generateKmeansClusters(series_scaled, numberOfClusters):
    cluster_count = math.ceil(math.sqrt(len(series_scaled))) 
    # A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN
    print("Training starting...")
    start = perf_counter()  
    km = TimeSeriesKMeans(n_clusters=numberOfClusters,max_iter=1000000, metric="euclidean")
    labels_knn = km.fit_predict(series_scaled)
    end  = perf_counter()  
    print("Training time :  " + str(end-start))
    #get clusters
    lst_knn = [[] for x in range(numberOfClusters)]
    for index,value in enumerate(labels_knn) : 
        lst_knn[value].append(index)
    print("Clusters : " +  str(lst_knn))


# def generateOpticsClusters(series_scaled,min_sample_value):
#     print("Training starting ...")
#     start = perf_counter()
#     clustering = OPTICS(min_samples=min_sample_value).fit(series_scaled)
#     end = perf_counter()
#     print("Training time :  " + str(end-start))
#
#     lstClusters = clustering.labels_.tolist()
#     #Get number of clusters
#     lstCl = []
#     for i in lstClusters:
#         if i not in lstCl:
#             lstCl.append(i)
#     #print(lstCl)
#     print("number of clusters : " + str(len(lstCl)))
#     numberOfClusters = len(lstCl)
#     lst_knn = [[] for x in range(numberOfClusters)]
#     for index,value in enumerate(clustering.labels_) :
#         lst_knn[value].append(index)
#     print("Clusters : " + str(lst_knn) )
#
#
#
# def generateSomClusters(plusVal,series_scaled,numberOfSeries,sigma_val, lr_val, iterations_val,neighborhood_function="gaussian",topology="rectangular",activation_distance="euclidean"):
#     sigma =  sigma_val
#     learning_rate =  lr_val
#     iterations = iterations_val
#     start = perf_counter()
#     som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(series_scaled))))
#
#     som_x = som_x - 1
#     print(som_x)
#     print(som_y)
#     #som_x = som_x + plusVal
#     #som_y = som_y + plusVal
#
#
#     som = MiniSom(som_x, som_y,len(series_scaled[0]), sigma=sigma, learning_rate = learning_rate,neighborhood_function=neighborhood_function,topology=topology,activation_distance=activation_distance)
#     som.random_weights_init(series_scaled)
#     #traning
#     som.train(series_scaled, iterations)
#     end = perf_counter()
#     print("Traning juste ended")
#     print("Training time :  " + str(end-start))
#
#     #clusters predictions
#     win_map = som.win_map(series_scaled)
#
#     #display histogram of clusters
#     cluster_c = []
#     cluster_n = []
#     for x in range(som_x):
#         for y in range(som_y):
#             cluster = (x,y)
#             if cluster in win_map.keys():
#                 cluster_c.append(len(win_map[cluster]))
#             else:
#                 cluster_c.append(0)
#             cluster_number = x*som_y+y+1
#             cluster_n.append(f"Cl {cluster_number}")
#
#     plt.figure(figsize=(25,5))
#     plt.title("Cluster Distribution for SOM")
#     plt.bar(cluster_n,cluster_c)
#     #plt.show()
#
#     numberOfClusters = len(cluster_n)
#     print("Number of clusters :  " + str(numberOfClusters))
#     #get clusters
#     cluster_map = []
#     #numberOfSeries = 325
#     series_labels = range(0,numberOfSeries)
#     for idx in range(len(series_scaled)):
#         print(idx)
#         winner_node = som.winner(series_scaled[idx])
#         #print(idx)
#         cluster_map.append((series_labels[idx],winner_node[0]*som_y+winner_node[1]+1))
#         #lst[(winner_node[0]*som_y+winner_node[1]+1) - 1].append(series_labels[idx])
#     res = pd.DataFrame(cluster_map,columns=["Series","Cluster"]).set_index("Series")
#     clusters_som = res.iloc[:,0].to_list()
#
#     lst_som = [[] for x in range(numberOfClusters)]
#     for index,value in enumerate(clusters_som) :
#         lst_som[value-1].append(index)
#     lst_som
#     print("Clusters : " + str(lst_som))
#







dataset  = importData("Datasets/national_illness.csv")
dataset = exstract_dataset(dataset).T #[num_series,samples]

scaled_series, scalers = normalizeDataset(dataset)
#scale_series: [num_series,samples]
scaled_series = scaled_series.T

# series_scaled format must be : (number of series,number of observations)
numberOfSeries = len(scaled_series)

#kmeans
determineKmeansClusters(scaled_series,numberOfSeries)
#generateKmeansClusters(scaled_series,100)

#optics 
#print( "scaled data" + str(scaled_series.shape))
#generateOpticsClusters(scaled_series,3)

#som 
#generateSomClusters(2,series_scaled,numberOfSeries,0.3,0.5,1000000,neighborhood_function='gaussian',topology='rectangular',activation_distance='euclidean')




