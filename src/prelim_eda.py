import numpy as np
import pandas as pd

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