import os
import csv
import argparse
import numpy as np
import nrrd
import utils
from visualize import render_cross_section


# use SAXM SAXMV for normal vector
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', type=str, required=True, help='SAXA')
    args = parser.parse_args()
    view = args.view
    if view not in ['SAXA', 'SAXM', 'SAXMV']:
        raise ValueError('only SAXA, SAXM, SAXMV')
    
    test_meta_path = 'data/meta/test/_TEST.txt'
    saxa_structs = utils.VIEW_STRUCTS['SAXA']
    saxm_structs = utils.VIEW_STRUCTS['SAXM']
    saxmv_structs = utils.VIEW_STRUCTS['SAXMV']
    save_dir = f'results/{view}_norm'

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
    saxa_evaluated_data = {}
    for struct in saxa_structs:
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
        saxa_evaluated_data[struct] = struct_dict
    
    saxm_evaluated_data = {}
    for struct in saxm_structs:
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
        saxm_evaluated_data[struct] = struct_dict

    saxmv_evaluated_data = {}
    for struct in saxmv_structs:
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
        saxmv_evaluated_data[struct] = struct_dict

    normal_error, centroid_error, normal_centroid_error, real_centroid_error, real_normal_centroid_error = [], [], [], [], []
    
    size = len(test_filenames)
    for idx, test_filename in enumerate(test_filenames):
        saxa_truth_xyz = []
        saxm_truth_xyz = []
        saxmv_truth_xyz = []

        saxa_pred_xyz = []
        saxm_pred_xyz = []
        saxmv_pred_xyz = []
        ratio = -1
        valid_structs = []

        for struct in saxa_structs:
            if test_filename in saxa_evaluated_data[struct]:
                row = saxa_evaluated_data[struct][test_filename]
                saxa_truth_xyz.append([row[0], row[1], row[2]])
                saxa_pred_xyz.append([row[3], row[4], row[5]])
                if view == 'SAXA':
                    ratio = row[6]
                    valid_structs.append(struct)

        for struct in saxm_structs:
            if test_filename in saxm_evaluated_data[struct]:
                row = saxm_evaluated_data[struct][test_filename]
                saxm_truth_xyz.append([row[0], row[1], row[2]])
                saxm_pred_xyz.append([row[3], row[4], row[5]])
                if view == 'SAXM':
                    ratio = row[6]
                    valid_structs.append(struct)

        for struct in saxmv_structs:
            if test_filename in saxmv_evaluated_data[struct]:
                row = saxmv_evaluated_data[struct][test_filename]
                saxmv_truth_xyz.append([row[0], row[1], row[2]])
                saxmv_pred_xyz.append([row[3], row[4], row[5]])
                if view == 'SAXMV':
                    ratio = row[6]
                    valid_structs.append(struct)

        if view == 'SAXA' and len(saxa_truth_xyz)<3:
            continue
        if view == 'SAXM' and len(saxm_truth_xyz)<3:
            continue
        if view == 'SAXMV' and len(saxmv_truth_xyz)<3:
            continue

        saxa_truth_xyz = np.array(saxa_truth_xyz)
        saxm_truth_xyz = np.array(saxm_truth_xyz)
        saxmv_truth_xyz = np.array(saxmv_truth_xyz)

        saxa_pred_xyz = np.array(saxa_pred_xyz)
        saxm_pred_xyz = np.array(saxm_pred_xyz)
        saxmv_pred_xyz = np.array(saxmv_pred_xyz)

        pred_xyz = None
        pred_centroid = None
        if view == 'SAXA':
            pred_xyz = saxa_pred_xyz
            pred_centroid = saxa_pred_xyz.mean(axis=0)
        elif view == 'SAXM':
            pred_xyz = saxm_pred_xyz
            pred_centroid = saxm_pred_xyz.mean(axis=0)
        elif view == 'SAXMV':
            pred_xyz = saxmv_pred_xyz
            pred_centroid = saxmv_pred_xyz.mean(axis=0)
        else:
            raise ValueError('Invalid view')
        
        coplanar_pred_xyz = []
        if len(saxa_pred_xyz) >= 3:
            shifted_saxa_pred_xyz = saxm_pred_xyz - saxm_pred_xyz.mean(axis=0)
            coplanar_pred_xyz.extend(shifted_saxa_pred_xyz)
        if len(saxm_pred_xyz) >= 3:
            shifted_saxm_pred_xyz = saxm_pred_xyz - saxm_pred_xyz.mean(axis=0)
            coplanar_pred_xyz.extend(shifted_saxm_pred_xyz)
        if len(saxmv_pred_xyz) >= 3:
            shifted_saxmv_pred_xyz = saxmv_pred_xyz - \
                saxmv_pred_xyz.mean(axis=0)
            coplanar_pred_xyz.extend(shifted_saxmv_pred_xyz)
            
        coplanar_pred_xyz = np.array(coplanar_pred_xyz)
        _, pred_normal = utils.fit_plane(coplanar_pred_xyz)

        # groundtruth SAXA needs to be handled differently due to colinearity
        if view == 'SAXA':
            saxm_not_found = False
            saxmv_not_found = False
            if len(saxm_truth_xyz) < 3:
                saxm_not_found = True
            if len(saxmv_truth_xyz) < 3:
                saxmv_not_found = True
            
            if saxm_not_found and saxmv_not_found:
                print('SAXM and SAXMV not found, use SAXA')
                truth_centroid, truth_normal = utils.fit_plane(saxa_truth_xyz)
            elif saxm_not_found:
                print('SAXM not found, use SAXMV')
                truth_centroid = saxa_truth_xyz.mean(axis=0)
                shifted_truth_xyz = saxa_truth_xyz - \
                    saxa_truth_xyz.mean(axis=0)
                shifted_saxmv_truth_xyz = saxmv_truth_xyz - \
                    saxmv_truth_xyz.mean(axis=0)
                coplanar_truth_xyz = np.concatenate(
                    (shifted_truth_xyz, shifted_saxmv_truth_xyz), axis=0)
                _, truth_normal = utils.fit_plane(coplanar_truth_xyz)
            elif saxmv_not_found:
                print('SAXMV not found, use SAXM')
                truth_centroid = saxa_truth_xyz.mean(axis=0)
                shifted_truth_xyz = saxa_truth_xyz - \
                    saxa_truth_xyz.mean(axis=0)
                shifted_saxm_truth_xyz = saxm_truth_xyz - \
                    saxm_truth_xyz.mean(axis=0)
                coplanar_truth_xyz = np.concatenate(
                    (shifted_truth_xyz, shifted_saxm_truth_xyz), axis=0)
                _, truth_normal = utils.fit_plane(coplanar_truth_xyz)
            else:
                truth_centroid = saxa_truth_xyz.mean(axis=0)
                shifted_truth_xyz = saxa_truth_xyz - \
                    saxa_truth_xyz.mean(axis=0)
                shifted_saxm_truth_xyz = saxm_truth_xyz - \
                    saxm_truth_xyz.mean(axis=0)
                shifted_saxmv_truth_xyz = saxmv_truth_xyz - \
                    saxmv_truth_xyz.mean(axis=0)
                coplanar_truth_xyz = np.concatenate(
                    (shifted_truth_xyz, shifted_saxm_truth_xyz, shifted_saxmv_truth_xyz), axis=0)
                _, truth_normal = utils.fit_plane(coplanar_truth_xyz)
        elif view == 'SAXM':
            truth_centroid, truth_normal = utils.fit_plane(saxm_truth_xyz)
        elif view == 'SAXMV':
            truth_centroid, truth_normal = utils.fit_plane(saxmv_truth_xyz)
        
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