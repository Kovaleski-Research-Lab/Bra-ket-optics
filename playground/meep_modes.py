
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('../')

from src.eig import *
from src.geometry import *
from src.utils import *


if __name__ == "__main__":

    # Let's load in the meep fields

    path_data = '../../../data/meep-dataset-v2/volumes'

    files = [f for f in os.listdir(path_data) if f.endswith('.pkl')]
    files.sort()
    files = [os.path.join(path_data, f) for f in files]

    # Load the files into an array
    fields = []
    for f in tqdm(files):
        with open(f, 'rb') as file:
            fields.append(pickle.load(file)[1.55])

    fields = np.array(fields)
    print(fields.shape)


    # Load in the eigen vectors
    eigen_vectors = pickle.load(open('../eigen_vectors_all.pkl', 'rb'))
    print(eigen_vectors.shape)
    input()



