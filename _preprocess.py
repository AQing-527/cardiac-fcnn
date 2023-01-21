import os
import nrrd
import csv
import numpy as np
from scipy import ndimage
import utils

RESIZE_OUT_SIZE = [128, 128, 128]
SKIP_SAVED_NRRD = False

raw_data_path = 'data/meta/4d_ijk/metadata.csv'

nrrd_save_dir = 'data/nrrd'
meta_save_dir = 'data/meta/3d_ijk'

data_dict = {}


def pad(data: np.ndarray):
    max_length = max(data.shape)
    out_size = [max_length, max_length, max_length]
    pad_width = np.array([[0, 0], [0, 0], [0, 0]])
    offsets = np.array([0, 0, 0])
    for d in range(3):
        if data.shape[d] > out_size[d]:
            start = (data.shape[d]-out_size[d])//2
            end = start+out_size[d]
            data = data.take(indices=range(start, end), axis=d)
            offsets[d] = -start
        else:
            before = (out_size[d]-data.shape[d])//2
            after = out_size[d]-data.shape[d]-before
            pad_width[d] = [before, after]
            offsets[d] = before
    return np.pad(data, pad_width, 'constant'), offsets, max_length


if __name__ == '__main__':
    csv_reader = csv.reader(open(raw_data_path, 'r'))
    csv_mat = []
    for row in csv_reader:
        if csv_reader.line_num == 1:
            continue
        csv_mat.append(row)

    idx = 0
    space_scales = np.array([-1, -1, -1])
    offsets = np.array([-1, -1, -1])
    resize_scale = -1
    new_round = True
    while idx < len(csv_mat):
        nrrd_path = csv_mat[idx][0]
        time_idx = int(csv_mat[idx][1])
        struct_idx = int(csv_mat[idx][2])
        view_name = utils.get_view_name_by_struct_id(struct_idx)
        # do not save the nrrd file if it existed already
        nrrd_filename = nrrd_path.split('/')[-1]
        filename_wo_ext = nrrd_filename.split('.')[0]
        
        new_round = True
        data_4d, header = nrrd.read(nrrd_path)
        space_scales = (header['space directions'][1][0],
                            header['space directions'][2][1],
                            header['space directions'][3][2])
        # Scaling and padding
        data_3d = data_4d[time_idx]
        # print(data_3d.shape, end=' ')
        data_3d_scaled = ndimage.zoom(data_3d, space_scales)
        # print(data_3d_scaled.shape, end=' ')
        data_3d_padded, offsets, max_length = pad(data_3d_scaled)
        # print(data_3d_padded.shape, end=' ')
        resize_scale = RESIZE_OUT_SIZE[0]/max_length
        data_3d_resized = ndimage.zoom(data_3d_padded, np.array(
        RESIZE_OUT_SIZE)/np.array([max_length, max_length, max_length]))
            
        # divide the nrrd file to patches
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    nrrd_patch_save_path = os.path.join(nrrd_save_dir, f'{view_name}_{filename_wo_ext}_{x}_{y}_{z}.nrrd')
                    nrrd_patch = data_3d_resized[64*x:64+64*x,64*y:64+64*y,64*z:64+64*z]
                    # print(nrrd_patch.shape) -> (64, 64, 64)
                    nrrd.write(nrrd_patch_save_path, nrrd_patch)

        # Write meta
        while (new_round or (csv_mat[idx][0]==csv_mat[idx-1][0] and utils.get_view_name_by_struct_id(int(csv_mat[idx][2]))==utils.get_view_name_by_struct_id(int(csv_mat[idx-1][2])))):
            nrrd_path = csv_mat[idx][0]
            time_idx = int(csv_mat[idx][1])
            struct_idx = int(csv_mat[idx][2])
            view_name = utils.get_view_name_by_struct_id(struct_idx)
            nrrd_filename = nrrd_path.split('/')[-1]
            filename_wo_ext = nrrd_filename.split('.')[0]
            i, j, k = float(csv_mat[idx][3]), float(csv_mat[idx][4]), float(csv_mat[idx][5])
            i, j, k = i*space_scales[0], j * \
                space_scales[1], k*space_scales[2]
            i, j, k = i+offsets[0], j+offsets[1], k+offsets[2]
            i, j, k = i*resize_scale, j*resize_scale, k*resize_scale
            meta_save_path = os.path.join(
                    meta_save_dir, str(struct_idx), 'metadata.csv')
            if not os.path.exists(meta_save_path):
                with open(meta_save_path, 'w') as meta_file:
                    csv_writer = csv.writer(meta_file)
                    csv_head = ['nrrd_patch_path', 'displacement_i', 'displacement_j', 'displacement_k', 'classifier']
                    csv_writer.writerow(csv_head)
                    for x in range(2):
                        for y in range(2):
                            for z in range(2):
                                nrrd_patch_save_path = os.path.join(nrrd_save_dir, f'{view_name}_{filename_wo_ext}_{x}_{y}_{z}.nrrd')
                                displacement_i = i - (32 + 64 * x)
                                displacement_j = j - (32 + 64 * y)
                                displacement_k = k - (32 + 64 * z)
                                classifier = 0
                                if i >= 64*x and i < 64+64*x and j>= 64*y and j < 64+64*y and k>= 64*z and k < 64+64*z:
                                    classifier = 1
                                data_row = [nrrd_patch_save_path, displacement_i, displacement_j, displacement_k, classifier]
                                csv_writer.writerow(data_row)
            
            else:
                with open(meta_save_path, 'a+') as meta_file:
                    csv_writer = csv.writer(meta_file)
                    for x in range(2):
                        for y in range(2):
                            for z in range(2):
                                nrrd_patch_save_path = os.path.join(nrrd_save_dir, f'{view_name}_{filename_wo_ext}_{x}_{y}_{z}.nrrd')
                                displacement_i = i - (32 + 64 * x)
                                displacement_j = j - (32 + 64 * y)
                                displacement_k = k - (32 + 64 * z)
                                classifier = 0
                                if i >= 64*x and i < 64+64*x and j>= 64*y and j < 64+64*y and k>= 64*z and k < 64+64*z:
                                    classifier = 1
                                data_row = [nrrd_patch_save_path, displacement_i, displacement_j, displacement_k, classifier]
                                csv_writer.writerow(data_row)
            print(f'{idx}/{len(csv_mat)}')
            idx += 1
            new_round = False
            if idx >= len(csv_mat):
                break
                