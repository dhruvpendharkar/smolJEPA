import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

def random_masking(image, patch_size, mask_ratio):
    b, c, h, w = image.shape()
    num_patches = (h // patch_size) * (w // patch_size)
    num_masked = int(num_patches * mask_ratio)
    indices = torch.randperm(num_paches)
    masked_indices = indices[:num_masked]
    visible_indices = indices[num_masked:]
    return masked_indices, visible_indices
