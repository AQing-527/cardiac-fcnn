import os
import csv
import nrrd
import numpy as np
from torch.utils.data import Dataset


class EchoData(Dataset):
    def __init__(self, meta_dir) -> None:
        super().__init__()
        self.meta_dir = meta_dir
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
        echo_data = nrrd.read(meta[0])[0]  # np.uint8
        displacement_vector = np.array([meta[1], meta[2], meta[3]])
        classifier = meta[4]
        return (echo_data, displacement_vector, classifier)
