# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 06:16:37 2020

@author: Johannes Uhl, Department of Geography, University of Colorado Boulder.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from sklearn.datasets import load_iris,load_diabetes,load_boston,fetch_california_housing

### dummy data:
#datadf = pd.DataFrame(load_iris().data,columns=load_iris().feature_names)
#datadf = pd.DataFrame(load_diabetes().data,columns=load_diabetes().feature_names)
datadf = pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
#datadf = pd.DataFrame(fetch_california_housing().data,columns=fetch_california_housing().feature_names)

variables=datadf.columns

### columns for color coding:
colorcode_upper = variables[0]
colorcode_lower = variables[2]

exclude_colorcoded=True ### will exclude the variables used for colorcoding from the scatterplot matrix.
use_ranks=True ### generate a matrix of QQ plots rather than scatterplots
transform_to_01 = False ### transforms each column to [0,1]
standardize = False ### standardize each column
fs=16 # font size

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = fs 

if transform_to_01:
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in variables:
        x = datadf[[col]].values.astype(float)
        datadf[col] = min_max_scaler.fit_transform(x) 

if standardize:
    min_max_scaler = preprocessing.StandardScaler()
    for col in variables:
        x = datadf[[col]].values.astype(float)
        datadf[col] = min_max_scaler.fit_transform(x) 
    
if exclude_colorcoded:
    if use_ranks:
        for col in datadf[variables]:
            datadf[col]=datadf[col].rank(pct=True)
    indarr = datadf[variables].drop(labels=[colorcode_upper,colorcode_lower],axis=1).values
else:
    if use_ranks:
        for col in datadf[variables]:
            datadf[col]=datadf[col].rank(pct=True)
    indarr = datadf[variables].values   
    
cmap1 = plt.cm.get_cmap('jet') #colorcoding for variable 1, upper triangle
cmapvals1 = 1-((np.max(indarr[:,np.argwhere(variables==colorcode_upper)[0][0]])-indarr[:,np.argwhere(variables==colorcode_upper)[0][0]])/float(np.max(indarr[:,np.argwhere(variables==colorcode_upper)[0][0]])-np.min(indarr[:,np.argwhere(variables==colorcode_upper)[0][0]]))) 
currcols1 = cmap1(cmapvals1) 

cmap2 = plt.cm.get_cmap('gist_rainbow') #colorcoding for variable 2, lower triangle
cmapvals2 = 1-((np.max(indarr[:,np.argwhere(variables==colorcode_lower)[0][0]])-indarr[:,np.argwhere(variables==colorcode_lower)[0][0]])/float(np.max(indarr[:,-1])-np.min(indarr[:,np.argwhere(variables==colorcode_lower)[0][0]]))) 
currcols2 = cmap2(cmapvals2) 

if exclude_colorcoded:
    variables=np.array([xx for xx in variables if xx not in [colorcode_upper,colorcode_lower]])

fig, axes = plt.subplots(len(variables), len(variables), figsize=(10, 10),sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.05)
fig.subplots_adjust(wspace=0.05)

var1count=0
for var1 in variables:
    var2count=0
    for var2 in variables:
        ax = axes[var1count,var2count]
        if var1count==var2count: #main diagonal
            ax = axes[var1count,var2count]
            ax.patch.set_facecolor('lightgray')
            
            if var1count==0 and var2count==0:                
                ax.set_title(var1,fontsize=fs)
                ax.set_ylabel(var2,fontsize=fs,rotation=90)
                
        else:
            if not var1count>var2count:  ## upper triangle, colorcode by first variable ####################
                print (var1count,var2count)
                print (var1,var2)
                var1vals = indarr[:,np.argwhere(variables==var1)[0][0]]
                var2vals =indarr[:,np.argwhere(variables==var2)[0][0]]
                ax.patch.set_facecolor('white')
                im = ax.scatter(x=var2vals, y=var1vals, s=2, color = currcols1, alpha=1)

                #ax.set_xlim([0,1])
                #ax.set_ylim([0,1]) 
                #ax.get_xaxis().set_ticks([0,1])
                #ax.get_yaxis().set_ticks([0.1])                     
                if var1count==0:
                    ax.set_title(var2,fontsize=fs)
                    if var2count==0: 
                        ax.set_ylabel(var1,fontsize=fs,rotation=90)  
                                                
            else: ## lower triangle, colorcode by second variable ####################
                
                ax = axes[var1count,var2count]
                print (var1,var2)
                var1vals = indarr[:,np.argwhere(variables==var1)[0][0]]
                var2vals =indarr[:,np.argwhere(variables==var2)[0][0]]
                ax.patch.set_facecolor('white')
     
                im = ax.scatter(x=var2vals, y=var1vals, s=2, color = currcols2, alpha=1)
               
                #ax.set_xlim([0,1])
                #ax.set_ylim([0,1]) 
                #ax.get_xaxis().set_ticks([0,1])
                #ax.get_yaxis().set_ticks([0,1])                
                if var2count==0:
                    ax.set_ylabel(var1,fontsize=fs,rotation=90)  
                    if var1count==0:
                        ax.set_title(var2,fontsize=fs)                    
        var2count+=1                                   
    var1count+=1

plt.xticks([], []) # remove ticks
plt.yticks([], [])
fig.savefig('scatterplot_matrix.png', dpi=600)  
