# Homework 2
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import itertools

import glob

# Example code provided
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
n_folds = 5

data = (M, L, n_folds)

def run (a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf if they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
def run(method=RandomForestClassifier, df = (), cvInfo = {}, hyperParameters = {}):
    metrics = dict()
    metrics['method'] = dict()
    metrics['method']['clf'] = method
    metrics['method']['hypers'] = hyperParameters
    metrics['maxes'] = dict()
    metrics['mins'] = dict()
    metrics['fold scores'] = dict()
    metrics['fold scores']['accuracy'] = list()
    metrics['fold scores']['precision score'] = list()
    metrics['fold scores']['f1 score'] = list()

    #Unpack data and create KFold and classifier method objects
    X, Y = df
    function, nSplits, shuffle = list(cvInfo.values())
    kf = function(n_splits=nSplits, shuffle = shuffle)
    method = method(**hyperParameters)
    
    #loop through the folds
    for foldNum, (trainIndex, testIndex) in enumerate(kf.split(X, Y)):
        
        #Fit the method to the training indices and create predictions
        method.fit(X[trainIndex], Y[trainIndex])
        pred = method.predict(X[testIndex])
        
        #Use the predictions to generate metric scores and store them separately from the final output
        metrics['fold scores']['accuracy'].append(round(accuracy_score(pred, Y[testIndex]),5))
        metrics['fold scores']['precision score'].append(round(precision_score(pred,Y[testIndex]),5))
        metrics['fold scores']['f1 score'].append(round(f1_score(pred, Y[testIndex]),5))
        
    #Store max values of metrics accross different folds of the same model
    metrics['maxes']['accuracy'] = max(metrics['fold scores']['accuracy'])
    metrics['maxes']['precision score'] = max(metrics['fold scores']['precision score'])
    metrics['maxes']['f1 score'] = max(metrics['fold scores']['f1 score'])
    
    #Store min values of metrics accross different folds of the same model
    metrics['mins']['accuracy'] = min(metrics['fold scores']['accuracy'])
    metrics['mins']['precision score'] = min(metrics['fold scores']['precision score'])
    metrics['mins']['f1 score'] = min(metrics['fold scores']['f1 score'])
    
    return metrics


# 2. Expand to include larger number of classifiers and hyperparameter settings
def hyperGrid(hypers = {}):
    #configure list of parameters represented
    paramList = list(hypers)
    
    #configure potential combinations of hyperparameter values
    combinations = list(itertools.product(*list(hypers.values())))
    
    #creating a dictionary to store each hyperparameter combination
    h = dict()
    
    # loop through the possible combinations
    for n in range(0,len(combinations)):
        #create a dictionary entry to store the individual combination of hyperparameter values
        h['hyper set '+str(n+1)] = dict()
        
        #creating a list of variables equal in length to the number of parameters (excluding values)
        #that are given in the input and then unpacking the hypers to the variables
        letters = list(map(chr, range(97, 97 + len(list(hypers.keys())))))
        letters = combinations[n]
        
        #storing the combination of values into the dictionary with its own unique identifier
        for param in range(len(paramList)):
            h['hyper set '+str(n+1)][paramList[param]] = letters[param]
            
    return h

def compareModels(modelDict = {}, df = (),):
    results = dict()
    results['score results'] = dict()
    results['score results']['accuracy'] = dict()
    results['score results']['precision score'] = dict()
    results['score results']['f1 score'] = dict()
    results['models'] = dict()
    results['models']['type'] = dict()
    results['models']['MwHP'] = dict()
    modelCount = 0
    for model in modelDict:
        
        #declare model object
        clf = models[model]['function']  
        
        #Calculate hyperparameter configurations based off of provided input
        h = hyperGrid(modelDict[model]['hypers'])
        
        #loop through hyperparameter configurations and generate a model for each one
        for pSets in range(len(h)):
                
            #Unpack hyper parameter set
            hypers = {**h[list(h.keys())[pSets]]}
                
            #notification of model being ran
            modelCount +=1
            print('Running model',modelCount,':', clf(**hypers))
            
            #Initiate model through run function
            modelResults = run(method = clf, df = df, cvInfo = models[model]['cvInfo'], hyperParameters = hypers)
            
            #Record max and min scores from resulting models
            for metric in ['maxes', 'mins']:
                for score in modelResults[metric]:
                    if metric not in results['score results'][score]:
                        results['score results'][score][metric] = dict()
                    results['score results'][score][metric]['model '+str(modelCount)] = modelResults[metric][score]
                    #results['models']['MwHP']['model '+str(modelCount)] = clf(**hypers)
                    results['models']['type']['model '+str(modelCount)] = model
        
        #Notify that model set has finished running
        print('...',model,'finished...')
    #notify that     
    print('...comparisons finished...')

    return results

# 3. Find some simple data
# I will use Glob and PD to create a dataframe by iterating through the tennis data obtained from
# https://archive.ics.uci.edu/ml/datasets/Tennis+Major+Tournament+Match+Statistics
# The original data downloaded contained some inconsistencies between the column names
# after trying to play around with automating the process to create a dataframe without Pandas,
# I decided to go ahead and just manually change the CSV column names to match
import glob
import pandas as pd

#I am not sure why the following code doesnt work
# #Importing all rows from all relevant files using Glob
# #Define the file path, an empty pd dataframe, and an empty list to store data
# filePath = '/Users/chadkwong/Desktop/ML2/Home2ork 2/TennisData'
# files = glob.glob(filePath + '/*.csv')
# print(files)
# #df = pd.DataFrame()
# data = list()
# #loop through the files and join them together into one dataset
# for file in files:
#     tempDF = pd.read_csv(file)
#     data.append(tempDF)
# #Combining everything into one dataframe
# df = pd.concat(data)

filePath = '/Users/chadkwong/Desktop/ML2/Homework 2/TennisData/'
file1 = 'AusOpen-men-2013.csv'
file2 = 'AusOpen-women-2013.csv'
file3 = 'FrenchOpen-men-2013.csv'
file4 = 'FrenchOpen-women-2013.csv'
file5 = 'USOpen-men-2013.csv'
file6 = 'USOpen-women-2013.csv'
file7 = 'Wimbledon-men-2013.csv'
file8 = 'Wimbledon-women-2013.csv'

files = [file1, file2, file3, file4, file5, file6, file7, file8]
data = list()
for file in files:
    data.append(pd.read_csv(filePath+ file))
df = pd.concat(data)

#Data Cleanup
#Printing out the percentage of missing values
naCounts = df.isnull().sum() * 100 / len(df)
print('Missing values: \n', naCounts,'\n\n')

#Dropping columns with high volume of NA values
#I also drop the player 1 and 2 columns because I am not interested in the player names
manyNACol = ['Player1', 'Player2', 'WNR.1', 'UFE.1', 'TPW.1', 'ST3.1', 'ST4.1', 'ST5.1',
             'WNR.2', 'UFE.2', 'TPW.2', 'ST3.2', 'ST4.2', 'ST5.2']
df.drop(columns = manyNACol, inplace=True)

#drop remaining rows containing na values after removing redundant columns
df.dropna(inplace=True)
target = df.pop('Result')
attributes = df

#create dictionary to store models to be ran with all of their hyper parameters and cross validation options set
# Hyper parameters need to be specified as a dictionary with each value formatted as a list
models = dict()

#These are universally used CV parameters
splits = 5
shuffle = False

#Random Forest Dictionary 
models['rf'] = dict()
models['rf']['function'] = RandomForestClassifier
models['rf']['hypers'] = dict()
models['rf']['hypers']['n_estimators']=[50,100]
models['rf']['hypers']['class_weight']=['balanced',None]
models['rf']['hypers']['min_impurity_decrease'] = [0.0, 1.0]
models['rf']['cvInfo'] = dict()
models['rf']['cvInfo']['cvFunction'] = KFold
models['rf']['cvInfo']['nSplits'] = splits
models['rf']['cvInfo']['shuffle'] = shuffle

#SVC Dictionary
models['svc'] = dict()
models['svc']['function'] = SVC
models['svc']['hypers'] = dict()
models['svc']['hypers']['C']=[1,2]
models['svc']['hypers']['class_weight']=['balanced',None]
models['svc']['hypers']['kernel'] = ['rbf','linear']
models['svc']['cvInfo'] = dict()
models['svc']['cvInfo']['cvFunction'] = KFold
models['svc']['cvInfo']['nSplits'] = splits
models['svc']['cvInfo']['shuffle'] = shuffle

# classification data
clusters = 1
samples = 1000
nClasses = 2
classificationData = make_classification(n_samples = samples,
                                         n_features = 100,
                                         n_informative = 80,
                                         n_redundant = 20,
                                         n_clusters_per_class = clusters,
                                         n_classes = nClasses)


data = (attributes.to_numpy(), target.to_numpy())

r = compareModels(models, data)


# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
def generatePlots(inputData, save = False):
    #Create a figure for displaying all plots at once
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    figure(figsize=(12, 12), dpi=100)

    #Store the figure axes in a list for convenience in plotting loop
    plts = [ax1, ax2, ax3, ax4, ax5, ax6]

    #create list to store sorted scores 
    sortedScore = list()

    #loop through each score metric and its respective max and min scores and sort by value
    for score in inputData['score results']:
        for metric in list(inputData['score results'][score].keys()):
            #store the model and its respective scores as dictionary items in a list
            sortedScore.append(sorted(inputData['score results'][score][metric].items(), key = lambda x:x[1]))

    #looping through the number of plots expected
    for pSet in range(len(plts)):   
        #configure the model number as the X axis to include all models calculated    
        modelNum = [x.split()[1] for (x,y) in sortedScore[pSet]]

        #configure the corresponding score to the Y axis to measure by
        score = [y for (x,y) in sortedScore[pSet]]

        #Display data as a scatter plot to convey independency between points in plot
        plts[pSet].scatter(modelNum, score)

    #Include this loop for plot customization
    for ax in fig.get_axes():
        ax.set_xticklabels(modelNum,)
    
    #
    if save ==True:
        fig.savefig('results.png')    
    elif save==False:
        plt.show()

#generatePlots(r, save=False)

# 5. Please set up your code to be run and save the results to the directory that its executed from
generatePlots(r, save = True)
with open('Model Comparison Results.json', 'w') as fp:
    json.dump(r, fp)


# 6. Investigate grid search function

