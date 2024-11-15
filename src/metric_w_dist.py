import numpy as np
import ot
import torch
import numpy as np
import random
from PIL import Image
from torchvision.transforms import v2, Compose, Resize, Normalize, ToTensor, RandomCrop

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)

class WDistImageLoaderSampler(Sampler):
    def __init__(self, pil_image, crop=False, device='cpu'):
        super(WDistImageLoaderSampler, self).__init__(device)
        self.crop = crop
        image = pil_image.convert("RGB")

        if self.crop:
            crop_size = min(image.size)
            image = v2.CenterCrop(crop_size)(image)
        
        transform = ToTensor()
        image = transform(image)
        
        self.image = image.to(self.device).permute([1,2,0]).reshape(-1,3)


    def sample(self, n_subsamples=1024):
        return self.image[np.random.randint(0,len(self.image),n_subsamples),:]

def compute_w_dist(pil_image_1, pil_image_2, n_subsamples=1024):
    xs = WDistImageLoaderSampler(pil_image_1).sample(n_subsamples)
    xt = WDistImageLoaderSampler(pil_image_2).sample(n_subsamples)
    
    a, b = np.ones((n_subsamples,)) / n_subsamples, np.ones((n_subsamples,)) / n_subsamples  # uniform distribution on samples

    # loss matrix
    M = np.array(ot.dist(xs, xt, metric='euclidean'))
    W = ot.emd2(a, b, M)
    return W