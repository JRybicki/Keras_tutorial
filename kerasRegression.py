import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras

#Check the tensorflow version
print(tf.__version__)

#File to  be read with structure of columns
csvData      = 'C:\\tensorflow\\auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

#Read the csv file
raw_dataset = pd.read_csv(csvData, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

#Copy the data into a dataset and remove the invalid (NAN) rows
dataset = raw_dataset.copy()
dataset = dataset.dropna()

#Encode the origin with a human redable value instead of ENUM
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

#Split the dataset into training and test data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Create plots using seaborn using pyplot to show
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
#plt.show()

#Show statistics on mean, std, etc
print(train_dataset.describe().transpose())

#Make a copy od the dataset
train_features = train_dataset.copy()
test_features  = test_dataset.copy()

#Select feature to train on, in this case MPG
train_labels = train_features.pop('MPG')
test_labels  = test_features.pop('MPG')