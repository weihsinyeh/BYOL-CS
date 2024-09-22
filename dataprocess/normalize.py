from torchvision.transforms import transforms
import torch
image_size = 128

def data_normalize():
    return transforms.Compose([ transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize( mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))])