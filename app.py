from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import json
import urllib2
from sklearn.cluster import KMeans
import numpy as np
from numpy import genfromtxt
import pandas as pd
import pdb
import csv
import io
import math
from sklearn.preprocessing import StandardScaler


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

        results = kmeans(input_data, factors_count)
        pca(input_data, factors_count)

        data = input_data
        kmeansData = kmeans(input_data, factors_count)
        pcaData = pca(input_data, factors_count)
        return render_template('index.html',**locals()) 

# pdb.set_trace()

if __name__ == "__main__":
    app.run()

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
    X = np.array(array).T
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
    for val in matrix_w:
        maxVal = np.amax(val)
        cluster_nums.append(np.nonzero(val == maxVal)[0][0])
    
    output = []
    for idx, val in enumerate(cluster_nums):
        output.append([val] + array[idx])

    return output