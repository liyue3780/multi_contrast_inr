from model import MLPv2, MLPv2WithEarlySeg
from dataset import MultiModalMultiSegDataset

dataset_class_map = {
    "MultiModalMultiSegDataset": MultiModalMultiSegDataset,
}

model_class_map = {
    "MLPv2WithEarlySeg": MLPv2WithEarlySeg,
    "MLPv2": MLPv2
}


