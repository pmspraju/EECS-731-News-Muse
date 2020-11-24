# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:47:16 2020

@author: pmspr
"""
###########################################
# Suppress matplotlib user warnings
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
# Display inline matplotlib plots with IPython
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')
###########################################

import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties

import pandas as pd
import numpy as np

def pca_results(good_data, pca):
    try:

    	# Dimension indexing
    	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    	# PCA components
    	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
    	components.index = dimensions
    
    	# PCA explained variance
    	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    	variance_ratios.index = dimensions
    
    	# Create a bar plot visualization
    	fig, ax = plt.subplots(figsize = (34,8))
        fontP = FontProperties()
        fontP.set_size('x-small')
    
    	# Plot the feature weights as a function of the components
    	components.plot(ax = ax, kind = 'bar');
    	ax.set_ylabel("Feature Weights")
    	ax.set_xticklabels(dimensions, rotation=0)
        ax.legend(loc="lower center", ncol=6, bbox_to_anchor=(0.6,0.5), bbox_transform=fig.transFigure, prop=fontP)
    
    	# Display the explained variance ratios
    	for i, ev in enumerate(pca.explained_variance_ratio_):
    		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))
        
    	# Return a concatenated DataFrame
    	return pd.concat([variance_ratios, components], axis = 1)
    
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def pcadim(df):
    try:
        # Plot the feature weights as a function of the components
        # Create a bar plot visualization
    	fig, ax = plt.subplots(figsize = (14,5))
    	df.plot(ax = ax, kind = 'bar');
    	ax.set_ylabel("Feature Weights")
    	ax.set_xticklabels(df.index.tolist(), fontsize=8, rotation=85)
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def biplot(good_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute
    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax