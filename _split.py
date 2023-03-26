import os
import csv
import numpy as np

CLEANUP_EXIST = True

all_dir = 'data/meta/3d_ijk'
test_meta_path = 'data/meta/test/_TEST.txt'
structs = ['20']

log_transformed = False

if log_transformed:
    train_dir = 'data/meta/log_transformed/train'
    val_dir = 'data/meta/log_transformed/val'
    test_dir = 'data/meta/log_transformed/test'
else:
    train_dir = 'data/meta/train'
    val_dir = 'data/meta/val'
    test_dir = 'data/meta/test'

if __name__ == '__main__':
    # Read predetermined test data
    test_filenames = set()
    with open(test_meta_path) as file:
        for line in file:
            test_filenames.add(line.strip('\n'))

    for struct in structs:
        all_struct_dir = os.path.join(all_dir, struct)
        train_struct_dir = os.path.join(train_dir, struct)
        val_struct_dir = os.path.join(val_dir, struct)
        test_struct_dir = os.path.join(test_dir, struct)
        os.makedirs(train_struct_dir, exist_ok=True)
        os.makedirs(val_struct_dir, exist_ok=True)
        os.makedirs(test_struct_dir, exist_ok=True)
        if CLEANUP_EXIST:
            os.system(f'rm -rf {train_struct_dir}/*')
            os.system(f'rm -rf {val_struct_dir}/*')
            os.system(f'rm -rf {test_struct_dir}/*')

        csv_reader = csv.reader(open(os.path.join(all_struct_dir, 'metadata.csv'), 'r'))
        csv_mat = []
        for row in csv_reader:
            if csv_reader.line_num == 1:
                continue
            csv_mat.append(row)

        test_meta = []
        train_val_meta = []
        for i in range(len(csv_mat)):
            nrrd = csv_mat[i][0].split('/')[-1][6:-11]
            if nrrd in test_filenames:
                test_meta.append(csv_mat[i])
            else:
                train_val_meta.append(csv_mat[i])
        print(len(test_meta))
        np.random.seed(4801)
        shuffled_meta = np.random.permutation(train_val_meta)
        train_meta = shuffled_meta[:8 * len(shuffled_meta) // 10]
        val_meta = shuffled_meta[8 * len(shuffled_meta) // 10:]
        print(len(train_meta))
        print(len(val_meta))

        # Record filenames
        train_meta_save_path = os.path.join(train_struct_dir, 'metadata.csv')
        val_meta_save_path = os.path.join(val_struct_dir, 'metadata.csv')
        test_meta_save_path = os.path.join(test_struct_dir, 'metadata.csv')
        with open(train_meta_save_path, 'w') as meta_file:
            csv_writer = csv.writer(meta_file)
            csv_head = ['nrrd_patch_path', 'displacement_i', 'displacement_j', 'displacement_k', 'classifier']
            csv_writer.writerow(csv_head)
            for row in train_meta:
                if log_transformed:
                    new_row = []
                    new_row.append(row[0])
                    for i in range(1,4):
                        if float(row[i])>=0:
                            log_num = np.log(float(row[i])+1)
                            new_row.append(log_num)
                        else:
                            log_num = -1 * np.log((-1*float(row[i]))+1)
                            new_row.append(log_num)
                    new_row.append(row[4])
                    csv_writer.writerow(new_row)
                else:
                    csv_writer.writerow(row)
        
        with open(val_meta_save_path, 'w') as meta_file:
            csv_writer = csv.writer(meta_file)
            csv_head = ['nrrd_patch_path', 'displacement_i', 'displacement_j', 'displacement_k', 'classifier']
            csv_writer.writerow(csv_head)
            for row in val_meta:
                if log_transformed:
                    new_row = []
                    new_row.append(row[0])
                    for i in range(1,4):
                        if float(row[i])>=0:
                            log_num = np.log(float(row[i])+1)
                            new_row.append(log_num)
                        else:
                            log_num = -1 * np.log((-1*float(row[i]))+1)
                            new_row.append(log_num)
                    new_row.append(row[4])
                    csv_writer.writerow(new_row)
                else:
                    csv_writer.writerow(row)

        with open(test_meta_save_path, 'w') as meta_file:
            csv_writer = csv.writer(meta_file)
            csv_head = ['nrrd_patch_path', 'displacement_i', 'displacement_j', 'displacement_k', 'classifier']
            csv_writer.writerow(csv_head)
            for row in test_meta:
                if log_transformed:
                    new_row = []
                    new_row.append(row[0])
                    for i in range(1,4):
                        if float(row[i])>=0:
                            log_num = np.log(float(row[i])+1)
                            new_row.append(log_num)
                        else:
                            log_num = -1 * np.log((-1*float(row[i]))+1)
                            new_row.append(log_num)
                    new_row.append(row[4])
                    csv_writer.writerow(new_row)
                else:
                    csv_writer.writerow(row)
            
