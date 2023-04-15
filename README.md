# cardiac-fcnn

This project adapts the [patch-based fully convolutional neural network] to detect 32 cardiac landmarks in 3D echocardiography images and then reconstructs 7 cross-section views of the heart based on the predicted landmark locations. 

## Usages
### 1. Install dependencies

```
conda env create -f env.yaml
conda activate comp4801
```

### 2. Place raw data

- Place raw `json` annotations into `data/raw/json/`.
- Place raw `nrrd` images into `data/raw/nrrd/`.

The directory tree should be like this:

```
cardiac-fcnn
├─ data
│  └─ raw
│     ├─ json
│     │  ├─ 2021-09-17-annotations
│     │  │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ERS_3DQ.json
│     │  │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ESW_3DQ.json
│     │  │  └─ ...
│     │  ├─ 2021-09-20-annotations
│     │  │  └─ ...
│     │  └─ ...
│     └─ nrrd
│        ├─ 2021-09-17-3d-nrrd
│        │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ERS_3DQ.seq.nrrd
│        │  ├─ PWHOR191529000T_17Sep2021_BPCZ8ESW_3DQ.seq.nrrd
│        │  └─ ...
│        ├─ 2021-09-20-3d-nrrd
│        │  └─ ...
│        └─ ...
└─ ...
```

### 3. Preprocess data

1. Parse all raw `json` annotations into `csv` files.

    ```
    python _parse_raw.py
    ```

    - The results will be saved in `data/meta/4d_ijk/metadata.csv`.

2. Preprocess data (extract 3D from 4D and resize); generate groundtruth regression and classification results; split the dataset into training, validation, and testing sets.

    ```
    python _preprocess.py
    python _split.py
    ```

    - `TEST.txt` is a list of fixed test filenames.   
    - The processed data will be saved in `data/nrrd/`; the dataset meta will be saved in `data/meta/train/$STRUCT/`, `data/meta/val/$STRUCT/`, `data/meta/train_val/$STRUCT/`, and `data/meta/test/$STRUCT/`, where `$STRUCT` is a list of numerical ids of landmarks.

### 4. Train the model

Train with `gpu-interactive`.

```
python train.py --config $CFG
```

- `$CFG` is the config file. Example config files are provided in `configs/`.
- Checkpoints will be saved in `pths/$STRUCT/`; logs will be saved in `logs/$STRUCT/`.

### 5. Evaluate the model

Calculate the Euclidean distance between the predicted and groundtruth landmark locations.

```
python evaluate.py --struct $STRUCT --pth_path $CKPT
```

- `$STRUCT` is the landmark id.
- `$CKPT` is the path to the `pth` file.
- The results will be saved in `evaluation/$STRUCT`.

### 6. Reconstruct and visualize the cross-section views

- For cross-section views except for SAXA, use the general SVD method.
    ```
    python recover.py --view $VIEW
    ```
    - `$VIEW` is is the view abbreviation.

- For SAXA, jointly use SAXA's, SAXM's and SAXMV's landmarks to predict the normal. 
    ```
    python recover_saxa.py --view "SAXA"
    ```

    - The results will be saved in `results/$VIEW/`.
    - `fit.csv` records the predicted centroid and normal vector of the cross-section plane.
    - `err.csv` records the distances from predicted landmarks to the predicted cross-section plane.
    - `images/` contains visualizations of the cross-section views.
