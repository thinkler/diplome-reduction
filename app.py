from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import json
import urllib2
from sklearn.cluster import KMeans
import numpy as np
# FLASK_APP=app.py FLASK_DEBUG=1 python -m flask run
# 
from numpy import genfromtxt
import pandas as pd
import pdb
import csv
import io

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
        
        results = cluster_analysis(input_data, int(request.form['fcount']))
        data = results[0]
        perfData = results[1]
        return render_template('index.html',**locals()) 

# pdb.set_trace()

if __name__ == "__main__":
    app.run()

def cluster_analysis(array, fcount):
    performed_data = array

    kmeans = KMeans(n_clusters = fcount)
    kmeans.fit(performed_data)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    output_data = np.c_[labels, performed_data]
    output_data = output_data[np.argsort(output_data[:,0])]
    
    return [array, output_data.tolist()]

