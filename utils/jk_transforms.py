import torch
import random

class AddGaussianNoise(object):
    
    def __init__(self, mean=0.0, std=0.01):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomHorizontalFlip2DTensor(object):
    """Horizontally flip the given 4D Torch Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, in_tensor):
        """
        Args:
            img (Torch Tensor): Image to be flipped.

        Returns:
            Torch Tensor: Randomly flipped image.
        """
        if random.random() < self.p:
            return in_tensor.transpose(1,2) 
        return in_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip2DTensor(object):
    """Vertically flip the given 4D Torch Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, in_tensor):
        """
        Args:
            img (Torch Tensor): 4D array to be flipped along the 3rd axis.

        Returns:
            Torch Tensor: Randomly flipped image.
        """
        if random.random() < self.p:
            return in_tensor.flip(1) 
        return in_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
if __name__ == '__main__':

    # Create a random torch tensor
    test_tensor = torch.rand(1,8,8)
    print('Test tesnsor:', test_tensor)
    
    # Generate the transformers
    test_crop = RandomCrop2DTensor(crop_size=3)
    test_horizontal = RandomHorizontalFlip2DTensor()
    test_vertical = RandomVerticalFlip2DTensor()
    
    # Get the test crop
    crop = test_crop(test_tensor)
    print('Crop:', crop)    
    
    # Test the flips
    horizontal = test_horizontal(crop)
    print('Horizontal flip:', horizontal)  
    
    vertical = test_vertical(horizontal)
    print('Vertical Flip:', vertical)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    