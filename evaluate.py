import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import EchoData
from models.fcnn import FCNN
import utils


if __name__ == '__main__':
    threshold = 0.6
    log_transformed = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--struct', type=str, default='20', help='struct id')
    parser.add_argument('--dataset', type=str, default='test', help='test')
    parser.add_argument('--pth_path', type=str, default='pths/20/transfer/200.pth',
                        help='pth path')
    parser.add_argument('--model_key', type=str,
                        default='model_state_dict', help='model key')
    args = parser.parse_args()
    struct = args.struct
    dataset = args.dataset
    pth_path = args.pth_path
    model_key = args.model_key
    if log_transformed:
        meta_dir = f'data/meta/log_transformed/{dataset}/{struct}'
    else:
        meta_dir = f'data/meta/{dataset}/{struct}'
    # ijk_dir = f'data/meta/3d_ijk/{view}'
    save_dir = f'evaluation/{struct}'
    view = utils.get_view_name_by_struct_id(int(struct))
    ratios = utils.read_ratio(f'data/meta/size/{view}.csv')

    time = utils.current_time()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'threshold-{threshold}'+'.csv')
    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        header = ['name', 'truth_x', 'truth_y', 'truth_z',
                  'pred_x', 'pred_y', 'pred_z', 'euclidean_distance', 'real_distance', 'ratio']
        writer.writerow(header)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EchoData(meta_dir, norm_echo=True, augmentation=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        drop_last=False, num_workers=4)

    model = FCNN().to(device)
    if model_key is None or model_key == '':
        model.load_state_dict(torch.load(
            pth_path, map_location=torch.device(device)))
    else:
        checkpoint = torch.load(pth_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint[model_key])
    model.eval()

    centers = []
    for x in range(2):
        for y in range(2):
                for z in range(2):
                    centers.append([(32 + 64 * x), (32 + 64 * y), (32 + 64 * z)])
    centers = np.array(centers, dtype=float)

    dists = []
    real_dists = []
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
    
            # dists = np.array(dists)

            with open(save_path, 'a+') as file:
                writer = csv.writer(file)
                print(f'[{batch:>3d}/{size:>3d}]', end=' ')
                ratio = ratios.get(filename[0])

                row = [f'{filename[0]}']
                row.extend(landmark)

                dist = np.sqrt(np.sum((pred_landmark-landmark)**2)).item()
                real_dist = dist/ratio
                dists.append(dist)
                real_dists.append(real_dist)

                row.extend(pred_landmark)
                row.extend([dist, real_dist, ratio])
                writer.writerow(row)
                print(dist)

    dists = np.array(dists)
    real_dists = np.array(real_dists)
    with open(save_path, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(['[median]', '', '', '', '', '', '',
                        np.median(dists), np.median(real_dists), ''])
        writer.writerow(['[mean]', '', '', '', '', '', '',
                        dists.mean(), real_dists.mean(), ''])
        writer.writerow(['[std]', '', '', '', '', '', '',
                        dists.std(), real_dists.std(), ''])
    print(f'[median] {np.median(dists)} {np.median(real_dists)}')
    print(f'[mean] {dists.mean()} {real_dists.mean()}')
    print(f'[std] {dists.std()} {real_dists.std()}')