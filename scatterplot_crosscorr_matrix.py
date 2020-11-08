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
import scipy.stats

from sklearn.datasets import load_iris,load_diabetes,load_boston,fetch_california_housing

### dummy data:
#datadf = pd.DataFrame(load_iris().data,columns=load_iris().feature_names)
#datadf = pd.DataFrame(load_diabetes().data,columns=load_diabetes().feature_names)
datadf = pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
#datadf = pd.DataFrame(fetch_california_housing().data,columns=fetch_california_housing().feature_names)

variables=datadf.columns

### column for color coding:
colorcode_scatter = variables[0]

exclude_colorcoded=False ### will exclude the variable used for colorcoding from the scatterplot / correlation matrix.
use_ranks=True ### generate a matrix of QQ plots rather than scatterplots
transform_to_01 = False ### transforms each column to [0,1]
standardize = False ### standardize each column
fs=12 # font size

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = fs 

### cross correlation matrix:
colorscheme = 'coolwarm'
cmap_corr = plt.cm.get_cmap(colorscheme)
cmapvals_corr = np.arange(0.0,1.0,1000)

crosscorrmat = np.empty((variables.shape[0],variables.shape[0]))
var1count=0
sorted_vars=[]
for var1 in datadf.columns:
    sorted_vars.append(var1)
    var2count=0
    for var2 in datadf.columns:
        crosscorr = scipy.stats.pearsonr(datadf[var1].values,datadf[var2].values)[0]
        crosscorrmat[var1count,var2count] = crosscorr                                    
        var2count+=1                                   
    var1count+=1    
crosscorrmat = np.nan_to_num(crosscorrmat)

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
    indarr = datadf[variables].drop(labels=[colorcode_scatter],axis=1).values
else:
    if use_ranks:
        for col in datadf[variables]:
            datadf[col]=datadf[col].rank(pct=True)
    indarr = datadf[variables].values   
    
cmap1 = matplotlib.cm.jet #colorcoding for variable
cmapvals1 = 1-((np.max(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]])-indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]])/float(np.max(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]])-np.min(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]]))) 
currcols1 = cmap1(cmapvals1) 

if exclude_colorcoded:
    variables=np.array([xx for xx in variables if xx not in [colorcode_scatter]])

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
                                                
            else: ## lower triangle, show cross correlation ####################

                ax = axes[var1count,var2count]
                crosscor = crosscorrmat[var1count,var2count]
                crosscor_scaled = (1+crosscor)/2.0
                ax.patch.set_facecolor(cmap_corr(crosscor_scaled))
                ax.annotate("%.2f" % crosscor,(np.mean(axes[var2count,var1count].get_xlim()),np.mean(axes[var2count,var1count].get_ylim())),ha='center', va = 'center',fontsize=fs)
                ax.set_xlim(axes[var2count,var1count].get_xlim())
                ax.set_ylim(axes[var2count,var1count].get_ylim())  
               
                if var2count==0:
                    ax.set_ylabel(var1,fontsize=fs,rotation=90)  
                    if var1count==0:
                        ax.set_title(var2,fontsize=fs)                    
        var2count+=1                                   
    var1count+=1

plt.xticks([], []) # remove ticks
plt.yticks([], [])
fig.savefig('scatterplot_matrix_w_crosscorr.jpg', dpi=300)  

###### create colorbars for legends:

fig,ax = plt.subplots(figsize=(2,4))
cmap = matplotlib.cm.coolwarm
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.8)
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical',ticks=[0,0.2,0.4,0.6,0.8])
cb1.ax.set_yticklabels(['-1.0','-0.5','0.0','0.5','1.0'])
ax.yaxis.set_ticks_position('left')    
plt.tight_layout()
fig.savefig('scatterplot_matrix_w_crosscorr_colorbar_crosscor.png', dpi=70)  

fig,ax = plt.subplots(figsize=(2,4))
cmap = cmap1
norm = matplotlib.colors.Normalize(vmin=np.min(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]]), vmax=np.max(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]]))
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
ax.yaxis.set_ticks_position('right')    
plt.tight_layout()
fig.savefig('scatterplot_matrix_w_crosscorr_colorbar_scatter.png', dpi=70)    