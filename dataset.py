from pathlib import Path
from typing import Tuple
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataset_utils import norm_grid, get_image_coordinate_grid_nib, get_seg_coordinate_grid_nib, pad_tensor_columns, convert_seg_to_continuous
from scipy.spatial import KDTree

class _BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, image_dir):
        super(_BaseDataset, self).__init__()
        self.image_dir = image_dir
        assert os.path.exists(image_dir), f"Image Directory does not exist: {image_dir}!"

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self) -> int:
        r"""Returns the number of coordinates stored in the dataset."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

class MultiModalDataset(_BaseDataset):
    r""" Dataset of view1/contrast1 and view2/contrast2 T2w image sequence of the same patient.
    These could be e.g. an view1 and view2 T2w brain image, an view1 and view2 spine image, etc.
    However, both images must be registered to one another - the daataset already assumes this.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    def __init__(self, image_dir: str="", name = "BrainLesionDataset",
                subject_id: str = "123456", 
                contrast1_LR_str: str= 'flair3d_LR', 
                contrast2_LR_str: str='dir_LR',
                transform = None, target_transform = None,
                config = None):
        super(MultiModalDataset, self).__init__(image_dir)
        self.config = config
        self.dataset_name = name
        self.subject_id = subject_id
        self.contrast1_LR_str = contrast1_LR_str
        self.contrast2_LR_str = contrast2_LR_str
        self.contrast1_LR_mask_str = contrast1_LR_str.replace("LR", "mask_LR")
        self.contrast2_LR_mask_str = contrast2_LR_str.replace("LR", "mask_LR")
        self.contrast1_GT_str = contrast1_LR_str.replace("_LR", "")
        self.contrast2_GT_str = contrast2_LR_str.replace("_LR", "")
        self.contrast1_GT_mask_str = "brainmask"
        self.contrast2_GT_mask_str = "brainmask"

        # Use SAVE_PATH from config if available, otherwise use current directory
        if self.config is not None and hasattr(self.config, 'SETTINGS') and hasattr(self.config.SETTINGS, 'SAVE_PATH'):
            base_path = self.config.SETTINGS.SAVE_PATH
        else:
            base_path = os.getcwd()
        
        dataset_filename = (
            f'{self.dataset_name}_'
            f'{self.subject_id}_'
            f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
            f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
            f'{self.contrast1_LR_mask_str}_{self.contrast2_LR_mask_str}_'
            f'{self.contrast1_GT_mask_str}_{self.contrast2_GT_mask_str}_'
            f'.pt'
        )
        self.dataset_name = os.path.join(base_path, 'preprocessed_data', dataset_filename)

        print(self.dataset_name)

        # we assume a BIDS-style formated directory

        files = sorted(list(Path(self.image_dir).rglob('*.nii.gz'))) 
        files = [str(x) for x in files]
        
        # Images in the output directory should be excluded
        out_img_dir = os.path.join(base_path, 'images')
        files = [k for k in files if out_img_dir not in k]

        # only keep NIFTIs that follow specific subject 
        files = [k for k in files if self.subject_id in k]
        # print(files)

        # flair3d and flair3d_LR or t1 and t1_LR
        self.gt_contrast1 = [x for x in files if self.contrast1_GT_str in x and self.contrast1_LR_str not in x and 'mask' not in x][0]
        self.gt_contrast2 = [x for x in files if self.contrast2_GT_str in x and self.contrast2_LR_str not in x and 'mask' not in x][0]
        self.lr_contrast1 = [x for x in files if self.contrast1_LR_str in x and 'mask' not in x][0]
        self.lr_contrast2 = [x for x in files if self.contrast2_LR_str in x and 'mask' not in x][0]
        self.lr_contrast1_mask = [x for x in files if self.contrast1_LR_mask_str in x and 'mask' in x][0]
        self.lr_contrast2_mask = [x for x in files if self.contrast2_LR_mask_str in x and 'mask' in x][0]
        self.gt_contrast1_mask = [x for x in files if self.contrast1_GT_mask_str in x and 'mask' in x][0]
        self.gt_contrast2_mask = [x for x in files if self.contrast2_GT_mask_str in x and 'mask' in x][0]

        if os.path.isfile(self.dataset_name):
            print("Dataset available.")
            dataset = torch.load(self.dataset_name)
            self.data = dataset["data"]
            self.label = dataset["label"]
            self.points = dataset["points"]
            self.mask = dataset["mask"]
            self.affine = dataset["affine"]
            self.dim = dataset["dim"]
            self.len = dataset["len"]
            self.gt_contrast1 = dataset["gt_contrast1"]
            self.gt_contrast2 = dataset["gt_contrast2"]
            self.gt_contrast1_mask = dataset["gt_contrast1_mask"]
            self.gt_contrast2_mask = dataset["gt_contrast2_mask"]
            self.coordinates = dataset["coordinates"]
            self.min_c = dataset["min_c"]
            self.max_c = dataset["max_c"]
            print("Skipped preprocessing.")

        else:
            self.len = 0
            self.data = []
            self.label = []
            self._process()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[dict, dict]:
        data = self.data[idx]
        label = self.label[idx]
        mask = self.mask[idx]
        return data, label, mask

    def get_intensities(self):
        return self.label
    
    def get_mask(self):
        return self.mask

    def get_coordinates(self):
        return self.coordinates

    def get_affine(self):
        return self.affine
    
    def get_dim(self):
        return self.dim
    
    def get_contrast1_gt(self):
        return self.gt_contrast1
           
    def get_contrast2_gt(self):
        return self.gt_contrast2
        
    def get_contrast2_gt_mask(self):
        return self.gt_contrast2_mask
    
    def get_contrast1_gt_mask(self):
        return self.gt_contrast1_mask

    def _process(self):

        # This allows the user to identify if all images/masks are correctly assigned.

        print(f"Using {self.lr_contrast1} as contrast1.")
        print(f"Using {self.lr_contrast2} as contrast2.")
        print(f"Using {self.lr_contrast1_mask} as contrast1 mask.")
        print(f"Using {self.lr_contrast2_mask} as contrast2 mask.")
        print(f"Using {self.gt_contrast1} as gt contrast1.")
        print(f"Using {self.gt_contrast2} as gt contrast2.")
        print(f"Using {self.gt_contrast1_mask} as gt contrast1 mask.")
        print(f"Using {self.gt_contrast2_mask} as gt contrast2 mask.")

        contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast1)))
        contrast2_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast2)))      
        contrast1_mask_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast1_mask)))
        contrast2_mask_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast2_mask)))
        data_contrast1 = contrast1_dict["coordinates"]
        data_contrast2 = contrast2_dict["coordinates"]

        min1, max1 = data_contrast1.min(), data_contrast1.max()
        min2, max2 = data_contrast2.min(), data_contrast2.max()
        self.min_c, self.max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))

        print(f'Min/Max of Contrast 1 {min1, max1}')
        print(f'Min/Max of Contrast 2 {min2, max2}')
        print(f'Global Min/Max of Contrasts {self.min_c, self.max_c}')

        data_contrast1 = norm_grid(data_contrast1, xmin=self.min_c, xmax=self.max_c)
        data_contrast2 = norm_grid(data_contrast2, xmin=self.min_c, xmax=self.max_c)
        labels_contrast1 = contrast1_dict["intensity_norm"]
        labels_contrast2 = contrast2_dict["intensity_norm"]        
        mask_contrast1 = contrast1_mask_dict["intensity_norm"].bool()
        mask_contrast2 = contrast2_mask_dict["intensity_norm"].bool()
        labels_contrast1_stack = torch.cat((torch.ones(labels_contrast1.shape)*-1, labels_contrast1), dim=1)
        labels_contrast2_stack = torch.cat((labels_contrast2, torch.ones(labels_contrast2.shape)*-1), dim=1) 

        # assemble the data and labels
        self.data = torch.cat((data_contrast1, data_contrast2), dim=0)
        self.label = torch.cat((labels_contrast1_stack, labels_contrast2_stack), dim=0)
        self.mask = torch.cat((mask_contrast1, mask_contrast2), dim=0)
        self.points = torch.cat((contrast1_dict["points"], contrast2_dict["points"]), dim=0)
        self.len = len(self.label)

        # store the GT images to compute SSIM and other metrics!
        gt_contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast1)))
        gt_contrast2_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast2)))
        self.gt_contrast1 = gt_contrast1_dict["intensity_norm"]
        self.gt_contrast2 = gt_contrast2_dict["intensity_norm"]
        self.gt_contrast1_mask = torch.tensor(nib.load(self.gt_contrast1_mask).get_fdata()).bool()
        self.gt_contrast2_mask = torch.tensor(nib.load(self.gt_contrast2_mask).get_fdata()).bool()

        # inference grid
        # self.coordinates = gt_contrast1_dict["coordinates_norm"]
        self.coordinates = norm_grid(gt_contrast1_dict["coordinates"], xmin=self.min_c, xmax=self.max_c)
        self.affine = gt_contrast1_dict["affine"]
        self.dim = gt_contrast1_dict["dim"]
        
        # store to avoid preprocessing
        dataset = {
            'len': self.len,
            'data': self.data,
            'mask': self.mask,
            'label': self.label,
            'points': self.points,
            'affine': self.affine,
            'gt_contrast1': self.gt_contrast1,
            'gt_contrast2': self.gt_contrast2,
            'gt_contrast1_mask': self.gt_contrast1_mask,
            'gt_contrast2_mask': self.gt_contrast2_mask,
            'dim': self.dim,
            'coordinates': self.coordinates,
            'min_c': self.min_c,
            'max_c': self.max_c
        }
        preprocessed_dir = os.path.dirname(self.dataset_name)
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir, exist_ok=True)
        torch.save(dataset, self.dataset_name)

class InferDataset(Dataset):
    def __init__(self, grid):
        super(InferDataset, self,).__init__()
        self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        data = self.grid[idx]
        return data


class MultiModalMultiSegDataset(MultiModalDataset):
    def __init__(self, image_dir = "",
                 name="BrainLesionDataset",
                 subject_id = "123456",
                 contrast1_LR_str = 'flair3d_LR',
                 contrast2_LR_str = 'dir_LR',
                 transform=None,
                 target_transform=None,
                 config=None):
        super().__init__(image_dir, name, subject_id, contrast1_LR_str, contrast2_LR_str, transform, target_transform, config)

        # set file names
        self.contrast1_LR_seg_str = contrast1_LR_str.replace("LR", "seg_LR")
        self.contrast2_LR_seg_str = contrast2_LR_str.replace("LR", "seg_LR")

        files = sorted(list(Path(self.image_dir).rglob('*.nii.gz')))
        files = [str(x) for x in files]
        files = [k for k in files if self.subject_id in k]

        self.lr_contrast1_seg = [x for x in files if self.contrast1_LR_seg_str in x and 'seg' in x][0]
        self.lr_contrast2_seg = [x for x in files if self.contrast2_LR_seg_str in x and 'seg' in x][0]

        self.lr_contrast1_seg, self.t1_continuous_label_dict = convert_seg_to_continuous(self.lr_contrast1_seg)
        self.lr_contrast2_seg, self.t2_continuous_label_dict = convert_seg_to_continuous(self.lr_contrast2_seg)

        # the number of label classes
        self.label_num = None

        # add segmentation labels
        self._process_seg()
    
    def get_continuous_label_dict(self):
        return self.t1_continuous_label_dict, self.t2_continuous_label_dict
    
    def get_label_num(self):
        return self.label_num
    
    def _process_seg(self):
        print(f"Using {self.lr_contrast1_seg} as lr contrast1 seg.")
        print(f"Using {self.lr_contrast2_seg} as lr contrast2 seg.")

        # read segmentation images as two other inputs (same level as other two input images)
        contrast1_seg_dict = get_seg_coordinate_grid_nib(nib.load(str(self.lr_contrast1_seg)))
        contrast2_seg_dict = get_seg_coordinate_grid_nib(nib.load(str(self.lr_contrast2_seg)))

        # ------- input: coordinate -------
        data_ct1_seg = contrast1_seg_dict["coordinates"]
        data_ct2_seg = contrast2_seg_dict["coordinates"]

        min1, max1 = data_ct1_seg.min(), data_ct1_seg.max()
        min2, max2 = data_ct2_seg.min(), data_ct2_seg.max()
        # min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2])) # TODO: min_c, max_c should come from the main image, not seg
        # print(f'Min/Max of Seg 1 {min1, max1}')
        # print(f'Min/Max of Seg 2 {min2, max2}')
        # print(f'Global Min/Max of Seg {min_c, max_c}')
        data_ct1_seg = norm_grid(data_ct1_seg, xmin=self.min_c, xmax=self.max_c)
        data_ct2_seg = norm_grid(data_ct2_seg, xmin=self.min_c, xmax=self.max_c)
        
        # Which label coordinates do we want to import? If t2_only, we can just ignore T1 labels
        if getattr(self.config.DATASET, "USED_SEG_TYPE", None) == 't2_only':
            data_seg = data_ct2_seg
            labels_seg = contrast2_seg_dict["label_one_hot"].to(self.label.dtype)
        else:
            data_seg = torch.cat((data_ct1_seg, data_ct2_seg), dim=0)
            labels_seg_ct1_one_hot = contrast1_seg_dict["label_one_hot"]
            labels_seg_ct2_one_hot = contrast2_seg_dict["label_one_hot"]
            if labels_seg_ct1_one_hot.shape[1] != labels_seg_ct2_one_hot.shape[1]:
                raise ValueError('ct1 and ct2 label must be in same column')
            labels_seg = torch.cat((labels_seg_ct1_one_hot, labels_seg_ct2_one_hot), dim=0).to(self.label.dtype)
                
        # Create a KDtree to map segmentation coordinates to the nearest image coordinates within tolerance
        n_hot = labels_seg.shape[1]  # number of one-hot classes
        kdtree = KDTree(self.data.numpy())
        dist, nearest_idx = kdtree.query(data_seg, distance_upper_bound=1e-5)
        p_match = dist != np.inf 
        
        print(f'Number of matches: {p_match.sum()} out of {len(data_seg)} segmentation points.')
        
        # For indices that didn't match, add their coordinates to the data
        if (~p_match).sum() > 0:
            self.data = torch.cat((self.data, data_seg[~p_match]), dim=0)
            self.label = torch.cat((self.label, torch.ones(data_seg[~p_match].shape[0], self.label.shape[1]) * -1), dim=0)
            self.mask = torch.cat((self.mask, torch.ones(data_seg[~p_match].shape[0], 1, dtype=torch.bool)), dim=0)
            
        # Extend the label array with the number of one-hot labels, filling in with -1 for now
        self.label = torch.cat((self.label, torch.ones(self.data.shape[0], n_hot) * -1), dim=1)

        # For labels that didn't match, fill in the one-hot labels at the end of the label array
        if (~p_match).sum() > 0:
            self.label[-data_seg[~p_match].shape[0]:, -n_hot:] = labels_seg[~p_match]
            
        # For labels that matched, fill in the one-hot labels at the correct indices
        if p_match.sum() > 0:
            nearest_idx = torch.tensor(nearest_idx, dtype=torch.int64)
            self.label[nearest_idx[p_match], -n_hot:] = labels_seg[p_match]
        
        # Print some statistics on the number of points with labels
        have_contrast1 = self.label[:, 0] != -1
        have_contrast2 = self.label[:, 1] != -1
        have_label = (self.label[:, -n_hot:] != -1).any(dim=1)
        print(f'Total number of points: {len(self.label)}')
        print(f'Number of points with contrast1: {have_contrast1.sum()}')
        print(f'Number of points with contrast2: {have_contrast2.sum()}')
        print(f'Number of points with label: {have_label.sum()}')
        print(f'Number of points with contrast1 and label: {(have_contrast1 & have_label).sum()}')
        print(f'Number of points with contrast2 and label: {(have_contrast2 & have_label).sum()}')
        print(f'Number of points with contrast1 and contrast2: {(have_contrast1 & have_contrast2).sum()}')        

        # seg label num
        self.label_num = contrast2_seg_dict['label_num']
        if getattr(self.config.DATASET, "USED_SEG_TYPE", None) != 't2_only' and contrast2_seg_dict['label_num'] != contrast1_seg_dict['label_num']:
            raise ValueError('two modalities label should be same')


if __name__ == '__main__':

    dataset = MultiModalDataset(
                image_dir='miccai',
                name='miccai_dataset',          
                )

    print("Passed.")
    
