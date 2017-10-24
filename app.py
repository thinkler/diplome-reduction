from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import json
import urllib2
from sklearn.cluster import KMeans
from minisom import MiniSom
import numpy as np
from numpy import genfromtxt
import pandas as pd
import pdb
import csv
import io
import math
from sklearn.preprocessing import StandardScaler
from numpy import *
from numpy.linalg import *
from numpy.random import *
from sklearn.decomposition import PCA

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
 
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html',**locals()) 
        
    if request.method == 'POST':
        f = request.files['myfile']
        if not f:
            return "No file"
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        input_data = []

        for row in csv_input:
            row = [float(i.replace(",",".")) for i in row]
            input_data.append(row)

        factors_count = int(request.form['fcount'])

        data = input_data

        true_cluster_nums = []        
        for row in data:
            true_cluster_nums.append(int(row[0]))
            del row[0]

        kmeansData = kmeans(data, factors_count)
        pcaData = pca(data, factors_count)
        somaData = soma(data, factors_count)
        mdsData = mds(data, factors_count)
        
        kmeansData = cluster_to_int(kmeansData)
        pcaData = cluster_to_int(pcaData)
        somaData = cluster_to_int(somaData)
        mdsData = cluster_to_int(mdsData)

        print('Matches with etalon:')
        print(calc_errors(true_cluster_nums, kmeansData, pcaData, somaData, mdsData))
        print('Sizes of clusters:')
        print(clust_size(true_cluster_nums, kmeansData, pcaData, somaData, mdsData))

        out = []
        for index, row in enumerate(data):
            out.append([int(true_cluster_nums[index])] + row)

        pureData = out

        return render_template('index.html',**locals()) 

# pdb.set_trace()

if __name__ == "__main__":
    app.run()

def calc_errors(ideal, kmeans, pca, soma, mds):
    mds_error = 0
    pca_error = 0
    soma_error = 0
    kmeans_error = 0
    for i, el in enumerate(ideal):
        if mds[i][0] != el:
            mds_error = mds_error + 1
        if pca[i][0] != el:
            pca_error = pca_error + 1
        if soma[i][0] != el:
            soma_error = soma_error + 1
        if kmeans[i][0] != el:
            kmeans_error = kmeans_error + 1

    return [mds_error, pca_error, soma_error, kmeans_error]

def clust_size(ideal, kmeans, pca, soma, mds):
    mds_count = {}
    pca_count = {}
    soma_count = {}
    kmeans_count = {}
    ideal_count = {}
    for i, el in enumerate(ideal):
        if el in ideal_count:
            ideal_count[el] += 1
        else:
            ideal_count[el] = 1

        if mds[i][0] in mds_count:
            mds_count[mds[i][0]] += 1
        else:
            mds_count[mds[i][0]] = 1
        
        if pca[i][0] in pca_count:
            pca_count[pca[i][0]] += 1
        else:
            pca_count[pca[i][0]] = 1
        
        if soma[i][0] in soma_count:
            soma_count[soma[i][0]] += 1
        else:
            soma_count[soma[i][0]] = 1
        
        if kmeans[i][0] in kmeans_count:
            kmeans_count[kmeans[i][0]] += 1
        else:
            kmeans_count[kmeans[i][0]] = 1

    mds_error = {}
    pca_error = {}
    soma_error = {}
    kmeans_error = {}

    for key in ideal_count:
        mds_error[key] = mds_count[key] - ideal_count[key]    
        pca_error[key] = pca_count[key] - ideal_count[key]    
        soma_error[key] = soma_count[key] - ideal_count[key]    
        kmeans_error[key] = kmeans_count[key] - ideal_count[key]    

    return [ [mds_count, pca_count, soma_count, kmeans_count, ideal_count], [mds_error, pca_error, soma_error, kmeans_error]]

def cluster_to_int(array):
    new_arr = []
    for row in array:
        n_row = []
        for ind, el in enumerate(row):
            if (ind == 0):
                n_row.append(int(el))
            else:
                n_row.append(el)
        new_arr.append(n_row)
    return new_arr

def kmeans(array, fcount):
    performed_data = array

    kmeans = KMeans(n_clusters = fcount)
    kmeans.fit(performed_data)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    output_data = np.c_[labels, performed_data]
    output_data = output_data[np.argsort(output_data[:,0])]

    return output_data.tolist()

def pca(array, fcount):
    X = np.array(array)
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    arr = []

    for i in range(fcount):
        arr.append(eig_pairs[i][1].reshape(len(eig_vals),1))

    matrix_w = np.hstack(arr)
    result = X_std.dot(matrix_w)


    cluster_nums = []
    for val in result:
        maxVal = np.amax(val)
        cluster_nums.append(np.nonzero(val == maxVal)[0][0])

    output = []
    for idx, val in enumerate(cluster_nums):
        output.append([val] + array[idx])

    return output

def soma(array, fcount):
    array = np.array(array)
    size = len(array[0])
    som = MiniSom(len(array), fcount, size, sigma=0.8, learning_rate=0.7) 
    som.random_weights_init(array)
    som.train_batch(array, 10)  
    
    results = []

    for val in array.tolist():
        results.append([som.winner(val)[1]] + val)

    return results

def norm(vec):
    return sqrt(sum(vec**2))

def mds(data, dimensions):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    data = np.array(data)

    arrSize = len(data)
    distance = zeros((arrSize,arrSize))



    for (i, pointi) in enumerate(data):
        for (j, pointj) in enumerate(data):
            distance[i,j] = norm(pointi - pointj)

    d = distance

    (n,n) = d.shape
    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = mat(mean(E,1))
    Es = mat(mean(E,0))

        # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = array(E - transpose(Er) - Es + mean(E))

    [U, S, V] = svd(F)

    Y = U * sqrt(S)

    cluster_nums = []
    for val in Y[:,0:dimensions]:
        maxVal = np.amax(val)
        cluster_nums.append(np.nonzero(val == maxVal)[0][0])

    output = []
    for idx, val in enumerate(cluster_nums):
        output.append([val] + data[idx].tolist())

    return output

    
