import os
import csv
import nrrd
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


class EchoData(Dataset):
    def __init__(self, meta_dir, norm_echo=True, augmentation=True) -> None:
        super().__init__()
        self.meta_dir = meta_dir
        self.norm_echo = norm_echo
        self.augmentation = augmentation
        self.meta_csv = os.listdir(meta_dir)[0]
        csv_reader = csv.reader(open(os.path.join(self.meta_dir, 'metadata.csv'), 'r'))
        csv_mat = []
        for row in csv_reader:
            if csv_reader.line_num == 1:
                continue
            csv_mat.append(row)
        self.size = len(csv_mat)
        self.metas = csv_mat

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        meta = self.metas[index]
        nrrd_filename = meta[0].split('/')[-1][6:-11]
        echo_data = nrrd.read(meta[0])[0]  # np.uint8
        
        if self.augmentation:
            zoom = np.random.uniform(0.9, 1.1)
            shift = np.random.randint(-5, 5, size=3)
            angles = (np.random.uniform(-15, 15),
                      np.random.uniform(-10, 10),
                      np.random.uniform(-10, 10))
            echo_data = self.augment(echo_data, zoom, shift, angles)

        echo_data = echo_data.astype(np.float64)  # np.uint8 -> np.float64
        
        if self.norm_echo:
            echo_data = self.normalize_echo(echo_data)

        if self.augmentation:
            multiply = np.random.uniform(0.75, 1.25)
            add = np.random.uniform(-0.25, 0.25)
            echo_data = self.adjust_intensity(echo_data, multiply, add)
        
        echo_data = torch.from_numpy(echo_data).float()
        echo_data = torch.unsqueeze(echo_data, dim=0)

        displacement_vector = np.array([meta[1], meta[2], meta[3]])
        displacement_vector = displacement_vector.astype(np.float64)
        displacement_vector = torch.from_numpy(displacement_vector).float()
        
        classifier = float(meta[4])
        classifier = torch.tensor(classifier).float()

        return (nrrd_filename, echo_data, displacement_vector, classifier)


    def augment(self, echo_data, zoom, shift, angles):
        size = echo_data.shape

        if zoom < 1.0:
            echo_data = self.scale(echo_data, zoom)
            echo_data = self.fit_size(echo_data, size)
            echo_data = self.rotate_and_translate(echo_data, angles, shift)
            echo_data = self.fit_size(echo_data, size)
        else:
            echo_data = self.rotate_and_translate(echo_data, angles, shift)
            echo_data = self.scale(echo_data, zoom)
            echo_data = self.fit_size(echo_data, size)
        return echo_data

    def fit_size(self, data, size):
        data_size = np.array(data.shape)
        size = np.array(size)
        # pad smaller dimensions
        padded_size = np.maximum(data_size, size)
        padded_data = np.zeros(padded_size, dtype=data.dtype)
        start = (padded_size-data_size)//2
        end = start+data_size
        padded_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = data
        # crop larger dimensions
        start = (padded_size-size)//2
        end = start+size
        return padded_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def rotate_and_translate(self, data, angles, shift):
        return self.translate(self.rotate(data, angles), shift)

    def translate(self, data, shift):
        return ndimage.shift(data, shift, cval=0.0)

    def rotate(self, data, angles):
        data = ndimage.rotate(
            data, angles[0], axes=(1, 0), reshape=True, cval=0.0)
        data = ndimage.rotate(
            data, angles[1], axes=(2, 1), reshape=True, cval=0.0)
        data = ndimage.rotate(
            data, angles[2], axes=(0, 2), reshape=True, cval=0.0)
        return data

    def scale(self, data, zoom):
        return ndimage.zoom(data, zoom, cval=0.0)

    def adjust_intensity(self, data, multiply, add):
        return (data*multiply)+add

    def normalize_echo(self, data):
        max_echo = data.max()
        min_echo = data.min()
        mean_echo = (max_echo+min_echo)/2.0
        scale_echo = (max_echo-min_echo)/2.0
        return (data-mean_echo)/scale_echo
