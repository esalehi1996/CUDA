# Highly parallelized K-means and Mergesort with CUDA

This repository contains a highly parallelized implementation of the K-means clustering algorithm in the source code kmeans.cu. To compile the code run the following command in terminal.

nvcc -arch=compute_id -rdc=true -lcurand kmeans.cu

** Please note that you should replace "id" with your device compute architecture number.

The compiled program first takes three numbers as input which are respectively the number of data samples, the dimension of data samples and finally the number of clusters. Next, all the data samples will be received from standard input. The program computes the clusters for each sample and prints them in order.

This repository also contains a highly parallelized implementation of the Merge Sort algorithm. The algorithm is implemented by dynamic parallelism which should be supported by the device GPU. To compile this code run the following command in terminal.

nvcc -arch=compute_id -rdc=true -lcurand merge.cu

** Similar to the previous case, please note that you should replace "id" with your device compute architecture number.

