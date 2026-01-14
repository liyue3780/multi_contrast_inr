## Single-subject Multi-contrast MRI Super-resolution via Implicit Neural Representations

[![DOI](https://img.shields.io/badge/arXiv-https%3A%2F%2Fdoi.org%2F10.48550%2FarXiv.2303.15065-B31B1B)](https://doi.org/10.48550/arXiv.2303.15065) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## What's New

This fork extends the original implementation with the following enhancements:

### 🆕 New Features

1. **Segmentation Support**
   - Added `MLPv2WithEarlySeg` model for joint multi-contrast super-resolution and segmentation
   - Added `MultiModalMultiSegDataset` for handling segmentation labels alongside image data
   - Supports one-hot encoded segmentation labels with configurable segmentation types
   - Automatic segmentation label conversion and processing utilities

2. **TensorBoard Integration**
   - Replaced Weights & Biases (wandb) with TensorBoard for experiment tracking
   - Logs training metrics, losses, and images during training
   - Use `--logging` flag to enable TensorBoard logging

3. **Dynamic Model and Dataset Selection**
   - Configurable model selection via `MODEL.MODEL_CLASS` in config files
   - Configurable dataset selection via `DATASET.DATASET_CLASS` in config files
   - Centralized registry system (`registry.py`) for easy extension

4. **Enhanced Configuration System**
   - Reorganized config structure with `SETTINGS.SAVE_PATH` for output directory
   - New config options for segmentation (`DATASET.USED_SEG_TYPE`, `DATASET.DATASET_CLASS`)
   - Template config file: `config/config_pmc_mlpv2_seg.yaml`

5. **Improved Output Management**
   - Saves reconstructed images and segmentation outputs every 10 epochs
   - Organized output directory structure using `SAVE_PATH` from config
   - Separate directories for model weights and output images

### 📝 Code Organization

- **`registry.py`**: Centralized mappings for models and datasets
- **`dataset_utils.py`**: Utility functions for segmentation processing (`convert_seg_to_continuous`, `get_seg_coordinate_grid_nib`, etc.)
- **`utils.py`**: General utilities including `my_softmax` function

## We expect subject scans to be aligned to the following format

### BRATS 2019
```
├── BraTS19_CBICA_AZH_1
│   ├── BraTS19_CBICA_AZH_1_brainmask.nii.gz
│   ├── BraTS19_CBICA_AZH_1_flair_LR.nii.gz
│   ├── BraTS19_CBICA_AZH_1_flair_mask_LR.nii.gz
│   ├── BraTS19_CBICA_AZH_1_flair.nii.gz
│   ├── BraTS19_CBICA_AZH_1_t1_LR.nii.gz
│   ├── BraTS19_CBICA_AZH_1_t1_mask_LR.nii.gz
│   ├── BraTS19_CBICA_AZH_1_t1.nii.gz
│   ├── BraTS19_CBICA_AZH_1_t2_LR.nii.gz
│   ├── BraTS19_CBICA_AZH_1_t2_mask_LR.nii.gz
│   └── BraTS19_CBICA_AZH_1_t2.nii.gz
```

### Segmentation Dataset Format (for MultiModalMultiSegDataset)

For segmentation tasks, your dataset should include segmentation files following this naming convention:
- `{contrast1_LR_str}_seg_LR.nii.gz` (e.g., `t1_LR_seg_LR.nii.gz`)
- `{contrast2_LR_str}_seg_LR.nii.gz` (e.g., `t2_LR_seg_LR.nii.gz`)

The segmentation files will be automatically processed and converted to continuous labels using the `picsl_c3d` Python package.

**Example: Dataset with Segmentation (T1/T2 with segmentation)**

```
├── subject_id_001
│   ├── subject_id_001_brainmask.nii.gz
│   ├── subject_id_001_t1_LR.nii.gz
│   ├── subject_id_001_t1_mask_LR.nii.gz
│   ├── subject_id_001_t1_LR_seg_LR.nii.gz
│   ├── subject_id_001_t1.nii.gz
│   ├── subject_id_001_t2_LR.nii.gz
│   ├── subject_id_001_t2_mask_LR.nii.gz
│   ├── subject_id_001_t2_LR_seg_LR.nii.gz
│   └── subject_id_001_t2.nii.gz
```

**Notes:**
- Mask files (`*_mask_LR.nii.gz`) are required for each contrast
- Segmentation files must be in the same directory as the corresponding image files
- The segmentation files should contain integer labels (will be automatically converted to continuous labels)
- If using `USED_SEG_TYPE: t2_only` in config, only `t2_LR_seg_LR.nii.gz` is required (t1 segmentation can be omitted or will be ignored)
- The segmentation files are processed automatically and converted labels are saved in a `tmp_continuous_labels/` subdirectory

## Requirements

We provide an environment file that can be used to setup a conda environment. As our implementation is purely based on PyTorch, Scikit-Learn, Numpy, Nibabel, Pyyaml and the lpips repo, it should be easily possible to use other (or older) versions of libraries and CUDA, and tailor the environment to your needs.

### Additional Dependencies for Segmentation

- **picsl_c3d**: Python bindings for Convert3D (used for segmentation label remapping)
  - Installation: `pip install picsl_c3d`
  - GitHub: [picsl_c3d](https://github.com/pyushkevich/c3d_python)

## Usage

### Basic Training

As we train on single subjects, we decided to integrate training and inference into one python file, `main.py`.
Essentially, we run inference after every run - to log the performance of the isotropically upsampled image.
Please feel free to modify the codebase according to your needs or application.

To run the code, please execute:

```bash
python3 main.py --logging --config configs/your_custom_config.yaml
```

### Training with Segmentation

To train a model with segmentation support:

1. **Prepare your config file** (see `config/config_pmc_mlpv2_seg.yaml` as a template):
   ```yaml
   DATASET:
     DATASET_CLASS: MultiModalMultiSegDataset
     USED_SEG_TYPE: t2_only  # Options: 't2_only' or omit for both contrasts
   
   MODEL:
     MODEL_CLASS: MLPv2WithEarlySeg
     USE_TWO_HEADS: true
   ```

2. **Run training**:
   ```bash
   python3 main.py --logging --config configs/config_pmc_mlpv2_seg.yaml
   ```

3. **View TensorBoard logs**:
   ```bash
   tensorboard --logdir <SAVE_PATH>/tensorboard_logs
   ```

### Configuration Files

We provide experiment configurations with the following convention:

```
├── config_brats_ctr1.yaml -> Single Contrast for Contrast 1 (i.e. one output channel)
├── config_brats_ctr2.yaml -> Single Contrast for Contrast 2 (i.e. one output channel)
├── config_brats_mlpv2.yaml -> Multi Contrast Model with Split_Head Architecture (best performing model)
├── config_brats.yaml -> Multi Contrast Model without Split_Head Architecture (vanilla MLP, ablation)
├── config_pmc_mlpv2_seg.yaml -> Multi Contrast Model with Segmentation Support (NEW)
```

### Key Configuration Options

#### Model Selection
- `MODEL.MODEL_CLASS`: Choose model architecture
  - `MLPv2`: Standard multi-contrast model
  - `MLPv2WithEarlySeg`: Multi-contrast model with early segmentation branch

#### Dataset Selection
- `DATASET.DATASET_CLASS`: Choose dataset class
  - `MultiModalDataset`: Standard multi-modal dataset (default)
  - `MultiModalMultiSegDataset`: Multi-modal dataset with segmentation labels

#### Segmentation Options
- `DATASET.USED_SEG_TYPE`: Segmentation type
  - `t2_only`: Use segmentation labels only from contrast2
  - Omit or set to other value: Use segmentation labels from both contrasts

#### Output Paths
- `SETTINGS.SAVE_PATH`: Base directory for saving models and outputs
  - Model weights: `<SAVE_PATH>/weights/`
  - Output images: `<SAVE_PATH>/images/`
  - TensorBoard logs: `<SAVE_PATH>/tensorboard_logs/`

### Command Line Arguments

- `--config`: Path to config file (required)
- `--logging`: Enable TensorBoard logging
- `--subject_id`: Override subject ID from config
- `--batch_size`: Override batch size from config
- `--epochs`: Override number of epochs from config
- `--lr`: Override learning rate from config

### Example: Training with Custom Settings

```bash
python3 main.py \
  --config configs/config_pmc_mlpv2_seg.yaml \
  --logging \
  --subject_id your_subject_id \
  --batch_size 2000 \
  --epochs 100 \
  --lr 0.0005
```

## Model Architectures

### MLPv2WithEarlySeg

The `MLPv2WithEarlySeg` model extends `MLPv2` with an early segmentation branch:

- **Architecture**: 
  - Shared encoder with early branching point
  - Two output heads for multi-contrast reconstruction (contrast1, contrast2)
  - One segmentation head branching from an intermediate layer
  - Output: `[contrast1, contrast2, segmentation]`

- **Loss Function**:
  - MSE loss for reconstruction (contrast1 and contrast2)
  - BCE loss for segmentation (foreground classes only)
  - Total loss: `L_reconstruction + L_segmentation`

- **Segmentation Output**:
  - Softmax activation applied to segmentation predictions
  - Only non-background classes (excluding -1 padding) are supervised
  - Supports configurable number of segmentation classes

## Output Files

The training process generates:

1. **Model Weights**: Saved in `<SAVE_PATH>/weights/`
   - Checkpoints saved every epoch
   - Best model based on validation metrics

2. **Reconstructed Images**: Saved in `<SAVE_PATH>/images/`
   - Contrast1 reconstruction (every 10 epochs)
   - Contrast2 reconstruction (every 10 epochs)
   - Segmentation output (every 10 epochs, if using segmentation model)

3. **TensorBoard Logs**: Saved in `<SAVE_PATH>/tensorboard_logs/`
   - Training/validation losses
   - Segmentation loss (if applicable)
   - Image visualizations

## Extending the Codebase

### Adding a New Model

1. Define your model class in `model.py`
2. Register it in `registry.py`:
   ```python
   model_class_map = {
       "YourModel": YourModelClass,
       ...
   }
   ```
3. Use it in your config: `MODEL.MODEL_CLASS: YourModel`

### Adding a New Dataset

1. Define your dataset class (optionally inheriting from `MultiModalDataset`)
2. Register it in `registry.py`:
   ```python
   dataset_class_map = {
       "YourDataset": YourDatasetClass,
       ...
   }
   ```
3. Use it in your config: `DATASET.DATASET_CLASS: YourDataset`

## Citation and Contribution

Please cite this work if any of our code or ideas are helpful for your research.

```
@inproceedings{mcginnisandshit2023single,
  title={Single-subject Multi-contrast MRI Super-resolution via Implicit Neural Representations},
  author={McGinnis, Julian and Shit, Suprosanna and Li, Hongwei Bran and Sideri-Lampretsa, Vasiliki and Graf, Robert and Dannecker, Maik and Pan, Jiazhen and Stolt-Ans{\'o}, Nil and M{\"u}hlau, Mark and Kirschke, Jan S and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={173--183},
  year={2023},
  organization={Springer}
}
```

## License

This project is licensed under CC BY-NC-SA 4.0.
