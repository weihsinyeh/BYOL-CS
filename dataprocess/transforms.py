from torchvision.transforms import transforms
import torch
image_size = 128
def data_argument():
    return transforms.Compose([ transforms.Resize((image_size, image_size)),
                                transforms.RandomApply( [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.3),
                                transforms.RandomGrayscale(p = 0.2),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply( [transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2),
                                transforms.RandomResizedCrop((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize( mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms     = args[0]
        self.random_flip    = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output