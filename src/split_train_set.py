import os
from random import sample
import shutil

def make_holdout_group(parent_dir,new_dir,size=0.1):
    '''
    Moves a subset of images to a new directory so that they can be labeled by hand.
    Creates new directory if it does not already exist.

    Inputs:
    parent_dir: Directory path where the main image set resides
    new_dir: New directory for subset
    size: Percentage of images to move from parent_dir to new_dir
    '''

    #Checks if new path exists, if it doesn't, makes new path

    if not os.path.exists(new_dir):
            os.mkdir(new_dir)

    #Creates a random sampling from parent set

    sample_dir = sample(os.listdir(parent_dir),int(len(os.listdir(parent_dir))*size))

    #Moves sample set to new directory

    for img in sample_dir:
        shutil.move(f'{parent_dir}/{img}',f'{new_dir}/{img}')

if __name__ == '__main__':

    list_features = ['background','bricks_masonry','characters','cutscene','design_patterns','earth_rocks_terrain','fx','metal','natural_patterns','object','portraits','sprite_menu','text','water','wood']

    for i,folder in enumerate(list_features):
        make_holdout_group(f'../data/textures/train/{list_features[i]}',f'../data/textures/holdout/{list_features[i]}',size=.15)
        make_holdout_group(f'../data/textures/train/{list_features[i]}',f'../data/textures/val/{list_features[i]}',size=.15)