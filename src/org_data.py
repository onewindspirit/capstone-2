import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import operator
import os
from random import sample
import shutil
import itertools

import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#from sklearn.dummy import DummyClassifier

from functools import partial
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
from tensorflow.keras.callbacks import TensorBoard 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from pipeline import ImagePipeline
from split_train_set import make_holdout_group

def list_shapes(data):
    '''
    Returns a dictionary of the amount of images that share shapes in a given dataset. 
    '''
    shapes={}
    for img in data:
        key = f'{img.shape[0]} x {img.shape[1]}'
        if key not in shapes:
            shapes[key] = 0
        shapes[key] += 1
    return sorted(shapes.items(),key=operator.itemgetter(1),reverse=True)

def filter_data(data,min_res=32,only_squares=True):#OLD,may not use
    '''
    Takes a set of image labels and returns a resized version based on minimum resolution and whether or not the images are square.
    
    Inputs:
        sub_dir: subdirectory of image labels to look at
        min_res: minimum hight or width for the data. Will remove all images with a smaller resolution and resize anything bigger to match.
        only_squares: if True, only includes images that have a square resolution in the final set
    Outputs:
        imgs: filtered and resized images
    '''
    imgs = []
    if only_squares == True:
        for img in data:
            if img.shape[0] == img.shape[1]:
                if img.shape[0]>=min_res & img.shape[1]>=min_res:
                    imgs.append(img)
    else:
        for img in data:
            if img.shape[0]>=min_res & img.shape[1]>=min_res:
                imgs.append(img)
            
    return imgs

def img_gen_train(train_dir,validation_dir,test_dir,target_size=(32,32),color_mode='rgba',class_mode='categorical',batch_size=32):

    train_datagen = ImageDataGenerator(
                        #rescale=1./255,
                        #shear_range=0.2,
                        #zoom_range=0.2,
                        #horizontal_flip=True
                        )

    validation_datagen = ImageDataGenerator(
                        #rescale=1./255
                        )

    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size=target_size,
                                    color_mode=color_mode,
                                    batch_size=batch_size,
                                    class_mode=class_mode,
                                    interpolation='nearest')#might try bilinear

    validation_generator = validation_datagen.flow_from_directory(
                                    validation_dir,
                                    target_size=target_size,
                                    color_mode=color_mode,
                                    batch_size=batch_size,
                                    class_mode=class_mode,
                                    interpolation='nearest')#might try bilinear

    test_generator = validation_datagen.flow_from_directory(
                                    test_dir,
                                    target_size=target_size,
                                    color_mode=color_mode,
                                    batch_size=batch_size,
                                    class_mode=class_mode,
                                    interpolation='nearest')#might try bilinear                                           

    return train_generator,validation_generator,test_generator

def build_baseline_model(n_features,input_shape=[32,32,4]):
    '''
    Builds a super basic model for establishing a baseline and making sure all the preprocessing is working smoothly. Only has 2 conv layers with pooling. Based on https://github.com/ageron/handson-ml2/blob/master/14_deep_computer_vision_with_cnns.ipynb

    Inputs:
    n_features: number of catagories from training set
    input_shape = shape of images in dataset [width,height,number of channels]

    Output:
    model: baseline model ready for training
    '''
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=7, input_shape=input_shape,activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=7, input_shape=input_shape,activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=n_features, activation='softmax')
    ])
    return model

def build_sequential_model(n_features,input_shape=[32,32,4]):
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=input_shape,activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=input_shape,activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=input_shape,activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=n_features, activation='softmax')
    ])   
    return model

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def build_resnet(n_features=10,input_shape=[32,32,4]):
    model = keras.models.Sequential()
    model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                            input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_features, activation="softmax"))
    return model

def model_compile_fit(model,train_generator,validation_generator,test_generator,callbacks,metrics=['accuracy'],n_epoch_steps=2000,n_epochs=50,validation_steps=800):

    '''Fits a given model and returns some typical stuff'''

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=metrics)

    history = model.fit(train_generator,
                        steps_per_epoch=n_epoch_steps,
                        epochs=n_epochs,
                        validation_data=validation_generator,validation_steps=validation_steps,
                        callbacks=callbacks,
                        verbose=1)
    
    predictions = model.predict(test_generator)

    return history,predictions

def image_plot(rows,batch,predictions,features,save=False,save_dir='',cols=32,figsize=(4,4)):  
    """ 
    Plots images with thier predicted class and actual class
    Used after model has been fit to see results of test data
    
    Parameters: 
        rows (int): number of rows for the images displayed
        batch (ImageDataGenerator): test batch of pictures that will be predicted in model
        model : the model training images have been trained on to be used to predict test images
        cols (int) : number of columns displayed ( set to batch size )
        features (list): list of features to show
    
    Returns: 
        plot: Plot of images with their predicted class and their actual class
    """
    fig, axs = plt.subplots(rows,cols,figsize=(cols * figsize[0],rows * figsize[1]))

    for i in range(rows):
        images, labels = next(batch)
        for j, pic in enumerate(images):
            title = 'Predicted:' + ' ' + features[list(predictions[j]).index(predictions[j].max())] + ' ' + '\n' + 'Actual:' + ' ' + features[list(labels[j]).index(1)] + '\n' + 'Confidence:' + str(predictions[j].max().round(2))
            if rows > 1:
                axs[i,j].imshow(pic.astype('uint8'))
                axs[i,j].set_title(title,color='blue')
                axs[i,j].axis('off')
                if features[list(predictions[j]).index(predictions[j].max())] != features[list(labels[j]).index(1)]:
                  axs[i,j].set_title(title,color='red')
            else:
                axs[j].imshow(pic.astype('uint8'))
                axs[j].set_title(title,color='blue')
                if features[list(predictions[j]).index(predictions[j].max())] != features[list(labels[j]).index(1)]:
                  axs[j].set_title(title,color='red')
                axs[j].axis('off')

    plt.tight_layout()
    plt.show()

    if save == True:
      plt.savefig(save_dir,dpi=150,transparent=True)

def build_cnf_matrix(predictions,generator):
  '''
  Returns a CMF matrix from predicted and actual labels for a prediction imnage set
  '''

  y_pred = np.argmax(predictions, axis=1)

  y_actual_lst = [(np.argmax(generator[i][1], axis = 1)) for i in range(len(generator.labels)//generator.batch_size + 1)]
  y_actual_arr = np.append(np.stack(y_actual_lst[:-1]).flatten(),y_actual_lst[-1])

  return np.round(confusion_matrix(y_actual_arr, y_pred),2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(10,10)):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  fig, ax = plt.subplots(figsize=figsize)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      cm = cm * 100
      print("\nNormalized confusion matrix")
  else:
      print('\nConfusion matrix, without normalization')
  print(cm)
  print ()

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=90)
  plt.yticks(tick_marks, classes)

  fmt = '.0f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

if __name__ == '__main__':
    
    train_dir = '../data/textures/train'
    validation_dir = '../data/textures/val'
    test_dir = '../data/texture/holdout'

    checkpoints_dir = '../callbacks/checkpoints'
    logs_dir = '../callbacks/tb_logs/'

    baseline_checkpoint_dir = '../callbacks/checkpoints/baseline'
    baseline_log_dir = '../callbacks/tb_logs/baseline'
    seq_checkpoint_dir = '../callbacks/checkpoints/seq'
    seq_log_dir = '../callbacks/tb_logs/seq'
    resnet_checkpoint_dir = '../callbacks/checkpoints/resnet'
    resnet_log_dir = '../callbacks/tb_logs/resnet'

    list_features = ['background','bricks_masonry','characters','cutscene','design_patterns','earth_rocks_terrain','fx','metal','natural_patterns','object','portraits','sprite_menu','text','water','wood']
    n_features = len(list_features)
    img_res = 32

    #callbacks
    baseline_cb_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    patience=5,
                                                    verbose=1)

    baseline_cb_tb_logs = tf.keras.callbacks.TensorBoard(log_dir=baseline_log_dir,histogram_freq=1)

    baseline_cb_model_cp = tf.keras.callbacks.ModelCheckpoint(
                                                    monitor='val_accuracy',
                                                    filepath=baseline_checkpoint_dir,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    verbose=1)

    baseline_callbacks = [baseline_cb_early_stop,baseline_cb_tb_logs,baseline_cb_model_cp]

    seq_cb_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=5,
                                                verbose=1)

    seq_cb_tb_logs = tf.keras.callbacks.TensorBoard(log_dir=seq_log_dir,histogram_freq=1)

    seq_cb_model_cp = tf.keras.callbacks.ModelCheckpoint(
                                                    monitor='val_accuracy',
                                                    filepath=seq_checkpoint_dir,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    verbose=1)

    seq_callbacks = [seq_cb_early_stop,seq_cb_tb_logs,seq_cb_model_cp]

    resnet_cb_early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=5,
                                                verbose=1)

    resnet_cb_tb_logs = tf.keras.callbacks.TensorBoard(log_dir=resnet_log_dir,histogram_freq=1)

    resnet_cb_model_cp = tf.keras.callbacks.ModelCheckpoint(
                                                    monitor='val_accuracy',
                                                    filepath=resnet_checkpoint_dir,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    verbose=1)

    resnet_callbacks = [resnet_cb_early_stop,resnet_cb_tb_logs,resnet_cb_model_cp]

    train_generator,validation_generator,test_generator = img_gen_train(train_dir,validation_dir,test_dir,target_size=(img_res,img_res),color_mode='rgba')

    #baseline_model = build_baseline_model(n_features,input_shape=[img_res,img_res,4])

    #baseline_history,baseline_predictions = model_compile_fit(baseline_model,train_generator,validation_generator,test_generator,baseline_callbacks,n_epoch_steps=80,n_epochs=10,validation_steps=10)

    #seq_model = build_sequential_model(n_features,input_shape=[img_res,img_res,4])
    #seq_model.summary()

    #seq_history,seq_predictions = model_compile_fit(seq_model,train_generator,validation_generator,seq_callbacks,n_epoch_steps=80,n_epochs=10,validation_steps=10)

    resnet_model = build_resnet(n_features)

    resnet_history,resnet_predictions = model_compile_fit(resnet_model,train_generator,validation_generator,test_generator,resnet_callbacks,n_epoch_steps=80,n_epochs=10,validation_steps=10)

    image_plot(31,test_generator,resnet_predictions,features=list_features,save=False,save_dir='../plots/image_plot',cols=img_res)

    cnf_matrix = build_cnf_matrix(resnet_predictions,test_generator)