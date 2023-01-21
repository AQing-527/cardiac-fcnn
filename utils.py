import json
import os
from datetime import datetime
from PIL import Image
import torch
import numpy as np

STRUCTS = ['A2C-LV apex', 'A4C-LV apex', 'A4C-TV tip', 'ALAX-LV apex', 'Anterior mitral annulus', 'Anterolateral mitral annulus',
           'Anterolateral papillary muscle', 'Aortic annulus', 'Center of AV', 'IAS', 'IVS', 'IW', 'Interventricular septum',
           'LV', 'Lateral mitral annulus', 'MV anterior leaflet  A2', 'MV anterior leaflet  A3', 'MV anterior leaflet A1',
           'MV posterior leaflet P1', 'MV posterior leaflet P2', 'MV posterior leaflet P3', 'MV tip', 'Medial mitral annulus',
           'PV tip', 'Posterior mitral annulus', 'Posteromedial mitral annulus', 'Posteromedial papillary muscle', 'RV', 'RV apex',
           'SAXA-LV apex', 'SAXB-TV tip', 'Tricuspid annulus.']

VIEWS = ['2 chamber view (A2C)', '4 chamber view (A4C)', 'Apical LV short-axis view (SAXA)', 'Basal short-axis view (SAXB)',
         'Long-axis view (ALAX)', 'MV short-axis view (SAXMV)', 'Mid LV short-axis view (SAXM)']

VIEWS_ABBR = ['A2C', 'A4C', 'SAXA', 'SAXB', 'ALAX', 'SAXMV', 'SAXM']

VIEW_STRUCTS = {
    'A2C': [0, 5, 25],
    'A4C': [1, 2, 14, 21, 22, 31],
    'SAXA': [12, 28, 29],
    'SAXB': [8, 9, 23, 30],
    'ALAX': [3, 4, 7, 24],
    'SAXMV': [15, 16, 17, 18, 19, 20],
    'SAXM': [6, 10, 11, 13, 26, 27]
}


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
        # globals().update(config)
    print(json.dumps(config, indent=2))
    return config


def update_config(path, config):
    with open(path, 'w') as json_file:
        json.dump(config, json_file, indent=4)


def current_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_struct_name(idx):
    return STRUCTS[idx]


def get_struct_idx(name):
    return STRUCTS.index(name)


def get_view_name(idx):
    return VIEWS[idx]


def get_view_abbr(idx):
    return VIEWS_ABBR[idx]


def get_view_index(name, type='abbr'):
    if type == 'name':
        return VIEWS.index(name)
    elif type == 'abbr':
        return VIEWS_ABBR.index(name)


def get_view_name_by_struct_id(struct_idx):
    for key in VIEW_STRUCTS:
        if struct_idx in VIEW_STRUCTS[key]:
            return key


def draw(data, filename, mode='clip'):
    if mode == 'clip':
        data[data < 0.0] = 0.0

    int_data = (((data - data.min()) / (data.max() - data.min()))
                * 255.9).astype(np.uint8)
    image = Image.fromarray(int_data)
    image.save(filename)
