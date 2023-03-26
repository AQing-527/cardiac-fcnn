import os
import csv
import argparse
import numpy as np
import nrrd
import utils
from visualize import render_cross_section


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, required=True, help='A2C')
    args = parser.parse_args()
    view = args.view

    test_meta_path = 'data/meta/test/_TEST.txt'
    structs = utils.VIEW_STRUCTS[view]
    save_dir = f'results/{view}'

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'fit.csv')
    error_path = os.path.join(save_dir, f'err.csv')
    image_path = os.path.join(save_dir, 'images')
    os.makedirs(image_path, exist_ok=True)

    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        header = ['name', 'p_centroid', 't_centroid', 'p_normal', 't_normal',
                  'normal_angle', 'centroid_dist', 'dist_along_t_normal',
                  'real_centroid_dist', 'real_dist_along_t_normal', 'ratio']
        writer.writerow(header)
    with open(error_path, 'w') as file:
        error_writer = csv.writer(file)
        header = ['name', 'struct', 'distance']
        error_writer.writerow(header)

    # Read predetermined test data
    test_filenames = []
    with open(test_meta_path) as file:
        for line in file:
            test_filenames.append(line.strip('\n'))

    # Read evaluated data
    evaluated_data = {}
    for struct in structs:
        struct_dict = {}
        csv_file = f'evaluation/{struct}/threshold-0.csv'
        reader = csv.reader(open(csv_file, 'r'))
        for row in reader:
            if reader.line_num == 1:
                continue
            elif row[1]=='':
                break
            else:
                struct_dict[row[0]] = [float(row[1]), float(row[2]), float(row[3]), 
                                       float(row[4]), float(row[5]), float(row[6]), 
                                       float(row[9])]
        evaluated_data[struct] = struct_dict

    normal_error, centroid_error, normal_centroid_error, real_centroid_error, real_normal_centroid_error = [], [], [], [], []
    
    size = len(test_filenames)
    for idx, test_filename in enumerate(test_filenames):
        truth_xyz = []
        pred_xyz = []
        ratio = -1
        valid_structs = []
        for struct in structs:
            if test_filename in evaluated_data[struct]:
                row = evaluated_data[struct][test_filename]
                truth_xyz.append([row[0], row[1], row[2]])
                pred_xyz.append([row[3], row[4], row[5]])
                ratio = row[6]
                valid_structs.append(struct)

        if len(truth_xyz) < 3:
            continue
        truth_xyz = np.array(truth_xyz)
        truth_centroid, truth_normal = utils.fit_plane(truth_xyz)
        
        pred_xyz = np.array(pred_xyz)
        pred_centroid, pred_normal = utils.fit_plane(pred_xyz)

        centroid_distance = np.sqrt(
                np.sum((pred_centroid-truth_centroid)**2))
        normal_angle = utils.angle_between(pred_normal, truth_normal)
        normal_centroid_distance = utils.distance_along_direction(
                truth_centroid, pred_centroid, truth_normal)
        real_centroid_distance = centroid_distance/ratio
        real_normal_centroid_distance = normal_centroid_distance/ratio

        with open(save_path, 'a+') as file:
            writer = csv.writer(file)
            data_row = [test_filename, pred_centroid, truth_centroid,
                        pred_normal, truth_normal, normal_angle,
                        centroid_distance, normal_centroid_distance,
                        real_centroid_distance, real_normal_centroid_distance, ratio]
            writer.writerow(data_row)

        error_distances = utils.distance_to_plane(
                pred_xyz, pred_centroid, pred_normal)**2
        for i, struct in enumerate(valid_structs):
            with open(error_path, 'a+') as file:
                error_writer = csv.writer(file)
                data_row = [test_filename, struct, error_distances[i]]
                error_writer.writerow(data_row)

        normal_error.append(normal_angle)
        centroid_error.append(centroid_distance)
        normal_centroid_error.append(normal_centroid_distance)
        real_centroid_error.append(real_centroid_distance)
        real_normal_centroid_error.append(real_normal_centroid_distance)

        nrrd_data = np.zeros((128,128,128))
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    nrrd_patch_path = f'data/nrrd/{view}_{test_filename}_{x}_{y}_{z}.nrrd'
                    nrrd_patch = nrrd.read(nrrd_patch_path)[0].astype(np.float64)
                    nrrd_data[64*x:64+64*x,64*y:64+64*y,64*z:64+64*z] = nrrd_patch
        pred_image = render_cross_section(
            nrrd_data, pred_centroid, pred_normal)
        truth_image = render_cross_section(
            nrrd_data, truth_centroid, truth_normal)
        utils.draw(pred_image, os.path.join(
            image_path, f'{test_filename}_pred.png'))
        utils.draw(truth_image, os.path.join(
            image_path, f'{test_filename}_truth.png'))

        print(f'[{idx:>3d}/{size:>3d}] {centroid_distance} {normal_angle}')

    centroid_error = np.array(centroid_error)
    normal_error = np.array(normal_error)
    normal_centroid_error = np.array(normal_centroid_error)
    real_centroid_error = np.array(real_centroid_error)
    real_normal_centroid_error = np.array(real_normal_centroid_error)
    with open(save_path, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(['[median]', '', '', '', '',
                         np.median(normal_error), np.median(centroid_error),
                         np.median(normal_centroid_error), np.median(
                             real_centroid_error),
                         np.median(real_normal_centroid_error), ''])
        writer.writerow(['[mean]', '', '', '', '',
                         normal_error.mean(), centroid_error.mean(),
                         normal_centroid_error.mean(), real_centroid_error.mean(),
                         real_normal_centroid_error.mean(), ''])
        writer.writerow(['[std]', '', '', '', '',
                         normal_error.std(), centroid_error.std(),
                         normal_centroid_error.std(), real_centroid_error.std(),
                         real_normal_centroid_error.std(), ''])

    print(f'[median] {np.median(centroid_error)} {np.median(normal_error)}')
    print(f'[mean] {centroid_error.mean()} {normal_error.mean()}')
    print(f'[std] {centroid_error.std()} {normal_error.std()}')