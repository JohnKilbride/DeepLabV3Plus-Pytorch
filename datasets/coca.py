import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
from glob import glob
import torch

from PIL import Image



def coca_cmap(N=3, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
    
class CocaDataset(torch.utils.data.Dataset):
    
    cmap = coca_cmap()
    def __init__(self, root, image_set='train', transform=None):
        '''
        Args:
            root_dir (string): Directory with all of the images
            transform (calllable, optional): Optional transforms that can be 
                applied to the images. 
        '''
        # Initalize the class attributes
        self.root = root
        self.transform = transform
        self.num_classes = 3
        self.paths = None
        self.image_set='train'
        
        # Get the file names
        self.__get_file_list()
        
        return
    
    def __get_file_list(self):
        '''Get a list of the tensors in the target directory.'''
        
        # Glob all of the tiffs in the root dir
        self.paths = glob(self.root + "/*" + self.image_set + "/*.pt")
        
        if len(self.paths) == 0:
            raise ValueError("Dataset found no `.pt` files in specified directory.")
        
        return None

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index of tensor to retrieve
        
        Returns:
            torch.tensor: DL Tensor [row, col, band]
        
        '''  
        # if the idx provided is a tensor convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load in the image
        tensor_path = os.path.join(self.root_dir, self.paths[idx])
        
        # Load the tensor
        tensor_2d = torch.load(tensor_path)
            
        # Apply the transform if needed
        if self.transform is not None: 
            tensor_2d = self.transform(tensor_2d)
            
        # create the input tensor and the target labels
        x = tensor_2d[0:-1,:,:]
        y = tensor_2d[-1:,:,:]
        
        return x, y.squeeze(0).long()

    def __len__(self):
        return len(self.paths)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
