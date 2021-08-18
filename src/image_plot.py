import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator 


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