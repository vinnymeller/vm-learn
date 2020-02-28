import os
import numpy as np
import matplotlib.pyplot as plt


data_path = os.path.join('martians')

def load_image(path, img_name):

    img = plt.imread(os.path.join(path, img_name))

    if len(img.shape) == 3:
        img=np.mean(img, axis=2)

    n_rows, n_cols = img.shape

    return img


def get_data_sets():

    filenames = os.listdir(data_path)
    img_names = [f for f in filenames if f[-4:] == '.jpg']

    def training_set():
        while True:
            cant_find = True
            while(cant_find):
                img = load_image(data_path, np.random.choice(img_names))
                if img.shape != (120, 135):
                    continue
                cant_find = False
            yield img

    def evaluation_set():
        while True:
            cant_find = True
            while (cant_find):
                img = load_image(data_path, np.random.choice(img_names))
                if img.shape != (120, 135):
                    continue
                cant_find = False
            yield img

    return training_set, evaluation_set