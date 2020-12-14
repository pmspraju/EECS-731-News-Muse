# EECS-731-News-Muse
EECS731 Semester Project

## Abstract:
  Our project will be focused on applying various data science approaches on fake news. Fake news is a falsified article or story intended to mislead the audience. People or online bots use manipulated news content as a tool to spread propaganda, influence a network of people, and to gain socio-political or economic benefits. Additionally, owing to its inherent dynamics, social media has become a fertile ground for spreading fake news.
  
  However, we believe that a piece of news is at its core, a piece of data. And, using a data science lifecycle effectively, we can identify fake news and formulate methods to determine its impact from the spatiotemporal and social metadata. At first, we can perform some exploratory data analysis to get an idea about the context or distribution of the content. Then, we can preprocess the data and work on identifying or generating suitable features from the news text and the metadata. Eventually, we can harness powerful classical and modern classification, clustering, and regression techniques to reveal information from the data that would otherwise be hidden initially. Finally, we can evaluate the performance of our data science pipeline both quantitatively and qualitativel
  
## Datasets:
1. Kaggle Fake News Dataset: https://www.kaggle.com/c/fake-news/data
2. FakeNewsNet (Twitter content): https://github.com/KaiDMML/FakeNewsNet
3. Fakeddit (Reddit content with text and images): https://github.com/entitize/Fakeddit

## Folder structure:
### data - This folder has the datsets that were used in this project
1. data\external\Fake-News-Dataset 		- Fake news dataset
2. data\external\OnlineNewsPopularity 	- Online news popularity dataset 
3. data\processed\kaggle_features		- Combined processed features dataset

### docs - This folder has the reports
1. EECS731_Proposal_FakeNews			- Initial project proposal
2. EECS 731_Project Presentation		- Project presentation
3. EECS 731_Project report_News Muse	- Project report 

### notebooks - This folder has the code in the form of Jupyter notebooks 
1. FakeNewsDetector 					- Fake new classifier
2. NewsPopularity						- Online news popularity classfication, PCA and clustering
3. Kaggle_Featureextractions_clustering	- Feature extration and combining two datasets 
4. NewsPopularity_Regression			- Online news popularity regression and domain adaptation

### src - This folder has the source code for all the python scripts, deployement and UI
1. src\scripts							- Python scripts 
2. src\deployment						- Deployed code for backend and React UI
3. src\features							- Python script for feature extraction 

### references - This folder has papers and other reference materials 

## Overview 
1. Performed Exploratory analysis and Feature Engineering on Fake news dataset
2. Performed Classification on the Fakenews dataset
3. Performed Exploratory analysis and Feature Engineering on Online news popularity dataset 
4. Performed Classification, Principal component analysis and Clustering on Online news popularity dataset 
5. Extracted common features and combined two datasets 
6. Performed Regression on Online news popularity dataset and Domain adaptation on the combined dataset. 

## Team members:
1. Ishrak Hayet
2. Sai Damaraju
3. Sushmitha Boddi Reddy
4. Madhu Peduri

## References:
1. K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. 
Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.

2. Transfer learning: domain adaptation by instance-reweighting -
https://johanndejong.wordpress.com/2017/10/15/transfer-learning-domainadaptation-by-instance-reweighting/