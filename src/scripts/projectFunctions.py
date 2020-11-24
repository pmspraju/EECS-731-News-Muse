# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
#import sys
import time
import pandas as pd
import numpy  as np
from scipy import stats
#import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import mixture

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn import metrics

from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
#nltk.download('punkt')

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#Function to explore the data
def exploreData(data):
    try:
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]    
          
           # Print the results
           print ("-----------------------------------------------------------------------")
           print ("Total number of records: {}".format(rows))
           print ("Total number of features: {}".format(cols))
           print ("-----------------------------------------------------------------------")
           
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def missingValues(data):
    try:
           # Total missing values
           mis_val = data.isnull().sum()
         
           # Percentage of missing values
           mis_val_percent = 100 * mis_val / len(data)
           
           # Make a table with the results
           mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
           
           # Rename the columns
           mis_val_table_ren_columns = mis_val_table.rename(
           columns = {0 : 'Missing Values', 1 : '% of Total Values'})
           mis_val_table_ren_columns.head(4 )
           # Sort the table by percentage of missing descending
           misVal = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                   '% of Total Values', ascending=False).round(1)
                     
           return misVal, mis_val_table_ren_columns

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def transformData(df):
    try:    
        #Get the list of columns having negative columns from describe
        nc = []
        for i in df.columns:
            if df[i].min() < 0:
                nc.append(i)
         
        target = df['shares']
        features_transform = pd.DataFrame(data = df)
        features_transform = features_transform.drop(columns=['shares'], axis=1)
        
        #Add constant value to make the negative values as positive.
        for i in nc:
            minv = features_transform[i].min()
            minv = minv * -1
            features_transform[i] = features_transform[i] + minv

        #Get the list of columns that are skew
        sc = []
        for i in features_transform.columns:
            if df[i].max() > 1:
                sc.append(i)
                
        #Scale the data to reduce the skewness 
#        features_transform[sc] = features_transform[sc].apply(lambda x: np.log(x + 1))
        scaler = MinMaxScaler() # default=(0, 1)
        features_transform[sc] = scaler.fit_transform(features_transform[sc])
        
        ind = np.where(target < 1400)
        target.iloc[ind] = 1
        
        ind = np.where(target >=1400)
        target.iloc[ind] = 0

        return features_transform, target
        
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#split the data in to train and test data
def splitData(features,target,testsize):
    try:
        # Split the 'features' and 'income' data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target, 
                                                    test_size = testsize, 
                                                    random_state = 1)

        # Show the results of the split
        print ("Features training set has {} samples.".format(X_train.shape[0]))
        print ("Features testing set has {} samples.".format(X_test.shape[0]))
        print ("Target training set has {} samples.".format(y_train.shape[0]))
        print ("Target testing set has {} samples.".format(y_test.shape[0]))
        print ("-----------------------------------------------------------------------")
        return X_train, X_test, y_train, y_test
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def corrPlot(corr):
    try:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    except Exception as ex:
        print ("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print (message)
           
def multinomialnb(X_train, X_test, y_train, y_test):
    try:
        #logic
        clf = MultinomialNB()
        params = {}

        scoring_fnc = make_scorer(fbeta_score,average='micro',beta=0.5)
        learner = GridSearchCV(clf,params,scoring=scoring_fnc)
        results = {}
         
        start_time = time.clock()
        grid = learner.fit(X_train,y_train)
        
        end_time = time.clock()
        results['train_time'] = end_time - start_time
        clf_fit_train = grid.best_estimator_
        start_time = time.clock()
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        results['f_train']   = fbeta_score(y_train, clf_predict_train, average='micro', beta=1)
        results['f_test']    = fbeta_score(y_test, clf_predict_test, average='micro', beta=1.5)
        
        return results,clf_fit_train      
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def svmClassifier(X_train, X_test, y_train, y_test):
    try:
        #Decision tree classifier
        #learner = DecisionTreeClassifier(criterion=method, max_depth=depth, random_state=1)
        clf = svm.SVC(random_state=0)
        params = {'gamma':[0.001]}
        #params = {'criterion':['gini','entropy'], 'max_depth' : np.array([6,7,8])}
         
        scoring_fnc = make_scorer(fbeta_score,average='micro',beta=0.5)
        learner = GridSearchCV(clf,params,scoring=scoring_fnc)
        results = {}
         
        start_time = time.clock()
        grid = learner.fit(X_train,y_train)
         
        end_time = time.clock()
        results['train_time'] = end_time - start_time
        clf_fit_train = grid.best_estimator_
        start_time = time.clock()
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
        
        return results,clf_fit_train      
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def randomForest(X_train, X_test, y_train, y_test):
    try:
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        #params = {}
        params = {'criterion':['gini','entropy'], 'max_depth' : np.array([6,7,8]),'random_state': [0]}
         
        scoring_fnc = make_scorer(fbeta_score,average='micro',beta=0.5)
        learner = GridSearchCV(clf,params,scoring=scoring_fnc)
        results = {}
         
        start_time = time.clock()
        grid = learner.fit(X_train,y_train)
         
        end_time = time.clock()
        results['train_time'] = end_time - start_time
        clf_fit_train = grid.best_estimator_
        start_time = time.clock()
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = accuracy_score(y_train, clf_predict_train)
        results['acc_test']  = accuracy_score(y_test, clf_predict_test)
         
        return results,clf_fit_train, 
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def pca(features,dim):
    try:
        #logic
        pca = PCA(n_components=dim)
        pca.fit(features)
        reduced_dim = pca.transform(features)
        
#        from pics import pca_results
#        _ = pca_results(features, pca)
        
        dlist = [];
        for i in range(dim):
            s = "D" + str(i)
            dlist.append(s)
        pca_comp = pd.DataFrame(pca.components_,columns=features.columns,index = dlist)
        #pca_comp.transpose().to_csv('test.csv')
        return reduced_dim, pca_comp.transpose(), pca
    
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def gclus(reduced_data,ic):
    try:
        cluster =  mixture.GaussianMixture(covariance_type='spherical', init_params='kmeans',
        max_iter=100, means_init=None, n_components=ic, n_init=1,
        precisions_init=None, random_state=None, reg_covar=1e-06,
        tol=0.001, verbose=0, verbose_interval=10, warm_start=False,
        weights_init=None).fit(reduced_data)
        
        pred = cluster.predict(reduced_data)
        
        centers = np.empty(shape=(cluster.n_components, reduced_data.shape[1]))
        for i in range(cluster.n_components):
            density = stats.multivariate_normal(cov=cluster.covariances_[i], mean=cluster.means_[i]).logpdf(reduced_data)
            centers[i, :] = reduced_data[np.argmax(density)]
        score = metrics.silhouette_score(reduced_data, pred, metric='euclidean')
        
        return cluster, centers,score
        
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def getCounts(flist,dat):
    try:
        #logic
        cols = []; counts = []; fake = [];
        for f in flist:
            ind = []
            ind = np.where((dat[f] == 1) & (dat['shares'] == 1))[0].tolist()
            #df1 = dat.loc[ind,['data_channel_is_lifestyle','shares']]
            cols.append(f)
            counts.append(len(ind))
            fake.append(False)
        
        for f in flist:
            ind = []
            ind = np.where((dat[f] == 1) & (dat['shares'] == 0))[0].tolist()
            #df1 = dat.loc[ind,['data_channel_is_lifestyle','shares']]
            cols.append(f)
            counts.append(len(ind))
            fake.append(True)
            
        df = pd.DataFrame(columns = ['feature','count','fake'])
        df['feature'] = cols
        df['count']= counts
        df['fake'] = fake 
        
        return df
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def get2Counts(flist,dat):
    try:
        #logic
        cols = []; counts = []; fake = [];
        for f in flist:
            ind = []
            ind = np.where((dat[f] != ' ') & (dat['shares'] == 1))[0].tolist()
            #df1 = dat.loc[ind,['data_channel_is_lifestyle','shares']]
            cols.append(f)
            counts.append(len(ind))
            fake.append(False)
        
        for f in flist:
            ind = []
            ind = np.where((dat[f] != ' ') & (dat['shares'] == 0))[0].tolist()
            #df1 = dat.loc[ind,['data_channel_is_lifestyle','shares']]
            cols.append(f)
            counts.append(len(ind))
            fake.append(True)
            
        df = pd.DataFrame(columns = ['feature','count','fake'])
        df['feature'] = cols
        df['count']= counts
        df['fake'] = fake 
        
        return df
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)