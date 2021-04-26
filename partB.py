import spacy
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import random
import numpy as np
import wget
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#load nlp model
nlp = spacy.load('en_core_web_sm')

#read file
def downloadData():
    url = 'https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/543ae83b0b093ab7ff4f2891ab7bccf6721d7e6d/datasets/train/SemEval2018-T3-train-taskB.txt'

    # Download the file (if we haven't already)
    if not os.path.exists('./SemEval2018-T3-train-taskB.txt'):
        wget.download(url, './SemEval2018-T3-train-taskB.txt')

downloadData()
file = './SemEval2018-T3-train-taskB.txt'
df = pd.read_csv(file, sep = '\t' )


#get label distribution
print('Label distribution')
print(df['Label'].value_counts())

print('Relative label distribution')
print(df['Label'].value_counts()/len(df))

print('Label examples')
for i in [0,1,2,3]:
    print(i)
    _df = df[df['Label'] == i]
    print(_df['Tweet text'].values[0])
#generate random predictions 
df['prediction_rand'] = [random.randint(0,3) for  i in range(len(df))]

print('Accuracy from random predictions', accuracy_score(df['Label'], df['prediction_rand']))

print('Classification report from random predictions')

#classification report with random predictions
print(classification_report(df['Label'], df['prediction_rand']))


print('Accuracy by class breakdown for random predictions')

cm = confusion_matrix(df['Label'], df['prediction_rand'])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(pd.DataFrame({'Labels': [0,1,2,3], 'Accuracy': cm.diagonal()}))

#Generate predictions with majority label
df['prediction_maj'] = [0]*len(df)

print('Accuracy from random predictions', accuracy_score(df['Label'], df['prediction_maj']))

print('Classification report from majority predictions')

#classification report with random predictions
print(classification_report(df['Label'], df['prediction_maj']))


print('Accuracy by class breakdown for majority predictions')

cm = confusion_matrix(df['Label'], df['prediction_maj'])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(pd.DataFrame({'Labels': [0,1,2,3], 'Accuracy': cm.diagonal()}))
