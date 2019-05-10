
import itertools
import numpy as np
import os
import pandas as pd, pdb
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import figure
from matplotlib.backends import backend_agg
import seaborn as sns

def top_n_accuracy(predictions, one_hot_labels, top_n):
    if predictions.shape[0] == 0:
        return 0
    #predictions: np array of shape (num_samples, num_classes) containing probability of each class for each sample
    #one_hot_labels: np array of shape (num_samples, num_classes) containing one-hot-labels for each sample
    #top_n: integer, calculate the accuracy of finding the correct label in the top_n predictions
    predicted_labels_top_n = np.argsort(predictions, axis=1)[:,-top_n:]
    true_labels = np.argmax(one_hot_labels,axis=1) #IMPORTANT: this assumes that there is only one label (not multilabel)
    arr = []
    for i in range(predictions.shape[0]):
        arr.append(np.any(predicted_labels_top_n[i,:] == true_labels[i]))
    return np.sum(arr)/len(arr)

def calibration_df(predicted_probability, true_labels_cat, interval=0.05):
    #IMPORTANT: I only calculate calibration numbers for the top predictions (whether the highest probability for each sample/row is calibrated)
    b_preds_max = predicted_probability.max(axis=1)
    bins = np.arange(0,1.06,interval)
    df = pd.DataFrame.from_dict({'bins':bins, 'num':np.repeat(0,bins.shape[0]), 'acc':np.repeat(0,bins.shape[0]),'conf': np.repeat(0,bins.shape[0])})
    for i in range(bins.shape[0]-1):
        inds = (b_preds_max>=bins[i]) & (b_preds_max<bins[i+1])
        df.acc.iloc[i] = np.around(top_n_accuracy(predicted_probability[inds],true_labels_cat[inds],1), decimals = 3)
        df.num.iloc[i]=sum(inds)
        df.conf.iloc[i] = np.around(sum(b_preds_max[inds])/sum(inds) , decimals=3)
    return df
