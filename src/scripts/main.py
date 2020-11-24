# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, missingValues

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Project\Repos\News Popularity\Git\EECS-731-Project-7\Data'
filename = "OnlineNewsPopularity.csv"
data = loadData(path,filename)
data.columns = data.columns.str.strip()

#Check the missing values
misVal, mis_val_table_ren_columns = missingValues(data)
print(mis_val_table_ren_columns.head(20))

#des = data.describe().transpose().to_csv('test.csv')

#Remove rows with missing target values
col = ['url']
data = data.drop(columns=col, axis=1)

from projectFunctions import exploreData, transformData, splitData
exploreData(data)

###
dat = data
ind = np.where(dat['shares'] < 1400)
dat['shares'].iloc[ind] = 1
ind = np.where(dat['shares'] >=1400)
dat['shares'].iloc[ind] = 0

flist = []
f1list = ['data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus',
         'data_channel_is_socmed','data_channel_is_tech','data_channel_is_world']
flist.append(f1list)
f2list = ['weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday',
         'weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday']
flist.append(f2list)
f3list = ['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']
flist.append(f3list)
f4list = ['kw_min_min','kw_max_min','kw_avg_min',
          'kw_min_max','kw_max_max','kw_avg_max',
          'kw_min_avg','kw_max_avg','kw_avg_avg']
flist.append(f4list)
from projectFunctions import getCounts, get2Counts

fig, ax = plt.subplots(2,2,figsize = (40,40))
fig.subplots_adjust(hspace=1.3, wspace=0.15)
for fl in range(len(flist)):
    if fl < 2:
        df = getCounts(flist[fl],dat)
        _ = sns.barplot(x='feature', y = 'count', hue='fake', data=df, ax = ax[0][fl])
        st = "Feature set " + str(fl+1)
        ax[0][fl].set_title(st,fontsize=14)
        plt.sca(ax[0][fl])
        plt.xticks(rotation=90)
    else:
        df = get2Counts(flist[fl],dat)
        _ = sns.barplot(x='feature', y = 'count', hue='fake', data=df, ax = ax[1][fl-2])
        st = "Feature set " + str(fl)
        ax[1][fl-2].set_title(st,fontsize=14)
        plt.sca(ax[1][fl-2])
        plt.xticks(rotation=90)
plt.suptitle("Feature distribution")
plt.show()

sns.lineplot(data=dat, x="year", y="passengers", hue="month")
###

data_raw = data
features, target = transformData(data_raw)
#des = features.describe().transpose().to_csv('test.csv')

X_train, X_test, y_train, y_test = splitData(features,target,0.3)

from projectFunctions import multinomialnb, svmClassifier, randomForest, pca, gclus

#results,learner = multinomialnb(X_train, X_test, y_train, y_test)
#
#print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
#print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
#print ("-----------------------------------------------------------------------")
#
#results,learner = svmClassifier(X_train, X_test, y_train, y_test)
#
#print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
#print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
#print ("-----------------------------------------------------------------------")

#results,learner = randomForest(X_train, X_test, y_train, y_test)
#featureImp = pd.DataFrame(columns=['feature','importance'])
#featureImp['feature'] = features.columns
#featureImp['importance'] = learner.feature_importances_
#featureImp = featureImp.sort_values(by='importance')
#print(featureImp)
#print ("-----------------------------------------------------------------------")
#print ("Times for Training, Prediction: %.5f, %.5f" %(results['train_time'], results['pred_time']))    
#print ("Accuracy for Training, Test sets: %.5f, %.5f" %(results['acc_train'], results['acc_test']))     
#print ("-----------------------------------------------------------------------")

#dim = 2
#reduced_f, pca_comp, pca = pca(features,dim)
#from pics import pcadim, biplot
#pcad = pca_comp['D0']
#pcad = pca_comp.iloc[:,0]
#pcadim(pcad)
# Create a DataFrame for the reduced data
#reduced_f = pd.DataFrame(reduced_f, columns = ['Dimension 1', 'Dimension 2'])
#biplot(features, reduced_f, pca)
#clus_df = pd.DataFrame(columns=['clusters','score'])
#clist = [30]; slist = [];
#for dim in clist:
#    cluster, centers, score = gclus(reduced_f,dim)
#    slist.append(score * 100)
#    print ("-----------------------------------------------------------------------")
#    print "silhouette score for GMM: {:.4f}".format(score)
#    print "Optimal number of components: {:.4f}".format(cluster.n_components)
#    print "number of centers: {:.4f}".format(len(centers))
#    print ("-----------------------------------------------------------------------")
#
#clus_df['clusters'] = clist
#clus_df['score'] = slist
#
#ax = sns.barplot(x="clusters", y="score", data=clus_df)
#ax.set_title('Clustering scores')

#data.to_csv('test.csv',index=False)