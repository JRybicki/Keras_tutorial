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

###############################
#MODEL SETUP
###############################

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
#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
#plt.show()

#Show statistics on mean, std, etc
#print(train_dataset.describe().transpose())

#Make a copy of the dataset
train_features = train_dataset.copy()
test_features  = test_dataset.copy()

#Select feature to train on, in this case MPG
train_labels = train_features.pop('MPG')
test_labels  = test_features.pop('MPG')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
#print(normalizer.mean.numpy())

#Show normalized values
#first = np.array(train_features[:1])
#with np.printoptions(precision=2, suppress=True):
  #print('First example:', first)
  #print()
  #print('Normalized:', normalizer(first).numpy())

###############################
#LINEAR REGRESSION
###############################
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    tf.keras.layers.Dense(units=1)
])

#horsepower_model.summary()

#Run the model on the first 10 horsepower values
horsepower_model.predict(horsepower[:10])
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

##time
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging, otherwise screen will spam since this is a small model
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

#Print data from the last few epochs
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

plot_loss(history)
#plt.show()