import os
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EchoData
from models.fcnn import FCNN
import utils

pth_path = 'pths/0/Plateau-0.5-10_2023-03-22-11-20-23/100.pth'
meta_dir = 'data/meta/test/0'

save_dir = 'res'

log_transformed = False

threshold = 0.7


def save_txt(text, file):
    f = open(file, 'a+')
    f.write(str(text)+'\n')
    f.close()


if __name__ == '__main__':
    time = utils.current_time()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, time+'.txt')
    open(save_path, 'w').close()
    save_txt(f'pth_path: {pth_path}', save_path)
    save_txt(f'meta_dir: {meta_dir}', save_path)
    save_txt(f'threshold: {threshold}', save_path)
    save_txt('', save_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EchoData(meta_dir, norm_echo=True, augmentation=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=4)

    model = FCNN().to(device)
    model.load_state_dict(torch.load(
        pth_path, map_location=torch.device(device))['model_state_dict'])
    model.eval()

    centers = []
    for x in range(2):
        for y in range(2):
                for z in range(2):
                    centers.append([(32 + 64 * x), (32 + 64 * y), (32 + 64 * z)])
    centers = np.array(centers, dtype=float)

    dists = []
    size = len(loader)
    with torch.no_grad():
        for batch, (filename, echo, displacement_vector, classifier) in enumerate(loader):
            echo, displacement_vector, classifier = echo.to(device), displacement_vector.to(device), classifier.to(device)
            pred = model(echo)
            pred_displacement = pred[0]
            pred_classifier = pred[1]
            pred_displacement = np.array(pred_displacement.cpu())
            pred_classifier = np.array(pred_classifier.cpu())
            displacement_vector = np.array(displacement_vector.cpu())
            save_txt(pred_displacement, save_path)
            save_txt(pred_classifier, save_path)
            save_txt(displacement_vector, save_path)
            save_txt(classifier, save_path)
            num = 0
            largest_classifier_idx = 0
            pred_landmark = np.array([0,0,0], dtype=float)
            for i in range(len(pred_classifier)):
                if pred_classifier[i][0]>=threshold:
                    num += 1
                    if log_transformed:
                        reverse = []
                        for j in range(len(pred_displacement[i])):
                            if pred_displacement[i, j] >= 0:
                                reverse.append(np.exp(pred_displacement[i, j])-1)
                            else:
                                reverse.append(-1*(np.exp(-1*(pred_displacement[i, j]))-1))
                        pred_landmark += (centers[i] + np.array(reverse))
                    else:
                        pred_landmark += (centers[i] + pred_displacement[i])
                if pred_classifier[i][0]>pred_classifier[largest_classifier_idx][0]:
                    largest_classifier_idx = i
            print(num)
            if (num>0):
                pred_landmark = pred_landmark / num
            else:
                if log_transformed:
                    pred_reverse = []
                    for j in range(len(pred_displacement[largest_classifier_idx])):
                        if pred_displacement[largest_classifier_idx, j] >= 0:
                            pred_reverse.append(np.exp(pred_displacement[largest_classifier_idx, j])-1)
                        else:
                            pred_reverse.append(-1*(np.exp(-1*(pred_displacement[largest_classifier_idx, j]))-1))
                    pred_landmark = centers[largest_classifier_idx] + np.array(pred_reverse)
                else:
                    pred_landmark = centers[largest_classifier_idx] + pred_displacement[largest_classifier_idx]
            
            if log_transformed:
                truth_reverse = []
                for j in range(len(displacement_vector[0])):
                    if displacement_vector[0, j] >= 0:
                        truth_reverse.append(np.exp(displacement_vector[0, j])-1)
                    else:
                        truth_reverse.append(-1*(np.exp(-1*(displacement_vector[0, j]))-1))
                landmark = centers[0] + np.array(truth_reverse)
            else:
                landmark = centers[0] + displacement_vector[0]

            dist = np.sqrt(np.sum((pred_landmark-landmark)**2))
            
            dists.append(dist)
            save_txt(f'[{batch:>3d}/{size:>3d}] {filename[0]} {dist}', save_path)
            save_txt(landmark, save_path)
            save_txt(pred_landmark, save_path)
            save_txt('', save_path)

            print(f'[{batch:>3d}/{size:>3d}] {dist}')
    
    dists = np.array(dists)
    save_txt(f'[median] {np.median(dists)}', save_path)
    print(f'[median] {np.median(dists)}')

    save_txt(f'[mean] {dists.mean()}', save_path)
    print(f'[mean] {dists.mean()}')

    save_txt(f'[std] {dists.std()}', save_path)
    print(f'[std] {dists.std()}')

    save_txt(f'[median] in mm {np.median(dists)*1.25}', save_path)
    print(f'[median] in mm {np.median(dists)*1.25}')

    save_txt(f'[mean] in mm {dists.mean()*1.25}', save_path)
    print(f'[mean] in mm {dists.mean()*1.25}')

    save_txt(f'[std] in mm {dists.std()*1.25}', save_path)
    print(f'[std] in mm {dists.std()*1.25}')
