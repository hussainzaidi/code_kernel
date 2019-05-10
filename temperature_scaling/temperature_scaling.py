__doc__="Load a trained model, replicate with an added temperature scaling layer before softmax, train replicated model on validation,\
compare calibration by calculating the expected calibration error of the original and the replicated model.\
Author: Hussain Zaidi"


import sys
sys.path.append("../../../")
import pandas as pd, numpy as np
import tensorflow as tf
import custom_layers as c_layers
import validation

hub_module = "https://tfhub.dev/google/universal-sentence-encoder/2"


test_file = "test_path.txt" #assumed to be tab delimited, 2 numeric columns ["Questions", "Labels"]
val_file = "val_path.txt"

load_model_file = 'model_weights.h5' #h5 file for a keras model 


testing = pd.read_csv("{}".format(test_file), delimiter="\t",header=None, names=["Question","Labels"])#give whatever names you like
n_classes = testing.Labels.max()+1 
X_test, y_test = testing.Question, testing.Labels
y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)

valid = pd.read_csv("{}".format(val_file), delimiter="\t",header=None, names=["Question","Labels"])
X_val, y_val = valid.Question, valid.Labels
y_val_cat = tf.keras.utils.to_categorical(y_val, n_classes)


hidden_units = [256,128,64,32]
embed_size=512

#define original model
input = tf.keras.layers.Input(shape=(1,), dtype = tf.string)
net = tf.keras.layers.Dense(hidden_units[0], activation='relu', trainable=False)(input)
for units in hidden_units[1:]:
    net = tf.keras.layers.Dense(units, activation='relu', trainable=False)(net)
#add softmax
preds = tf.keras.layers.Dense(n_classes, activation='softmax', trainable=False)(net)
model = tf.keras.Model(inputs=input, outputs=preds)
model.load_weights(load_model_file)

#new_model replicates old model with an added scaling layer
input = tf.keras.layers.Input(shape=(1,), dtype = tf.string)
net = tf.keras.layers.Dense(hidden_units[0], activation='relu', trainable=False)(input)
for units in hidden_units[1:]:
    net = tf.keras.layers.Dense(units, activation='relu', trainable=False)(net)
#add a scaled layer
net = c_layers.Linear()(net)
#add softmax
preds = tf.keras.layers.Dense(n_classes, activation='softmax', trainable=False)(net)
new_model = tf.keras.Model(inputs=input, outputs=preds)
new_model.compile(optimizer=tf.train.AdagradOptimizer(learning_rate=0.3),
            loss='categorical_crossentropy', metrics=[tf.keras.metrics.categorical_accuracy])
#new_model has only 1 trainable parameter
#set the weights layer-by-layer
for i,layer in enumerate(model.layers):
    if i==(len(model.layers)-1):
        break
    if len(layer.get_weights())>0:
        new_model.layers[i].set_weights(layer.get_weights())
new_model.layers[-1].set_weights(model.layers[-1].get_weights())

#fit on validation data
new_model.fit(X_val, y_val_cat, epochs=100)
#parameter = 0.428, accuracy = 0.72

#predict
calibrated_preds = new_model.predict(X_test)
uncalibrated_preds = model.predict(X_test)

calibrated_df = validation.calibration_df(calibrated_preds, y_test_cat)
uncalibrated_df = validation.calibration_df(uncalibrated_preds, y_test_cat)
expected_calibrated_error = np.sum(abs(calibrated_df.acc - calibrated_df.conf)*calibrated_df.num)/sum(calibrated_df.num)
expected_uncalibrated_error = np.sum(abs(uncalibrated_df.acc - uncalibrated_df.conf)*uncalibrated_df.num)/sum(uncalibrated_df.num)


