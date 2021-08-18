import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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