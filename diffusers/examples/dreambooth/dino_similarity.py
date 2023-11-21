import os
from PIL import Image
import torch
import torch.nn as nn
import torch
from torchvision import transforms, datasets
from transformers import AutoImageProcessor, AutoModel


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, resolution):
        self.folder_path = folder_path
        self.images = os.listdir(folder_path)

        self.transform = transforms.Compose([
                                transforms.Resize(resolution),
                                transforms.CenterCrop((resolution, resolution)),
                                transforms.ToTensor()
                               ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.images[idx])
        image = Image.open(img_name)

        image = self.transform(image)

        return image

def img_folder_to_batch(folder, resolution):
    dataset = CustomImageDataset(folder, resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(dataloader))

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class DinoSimilarity:

    def __init__(self, resolution=None):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base') 
        self.resolution = resolution   

    def run_similarity(self, folder1, folder2):

        batch1 = img_folder_to_batch(folder1, self.resolution)
        batch2 = img_folder_to_batch(folder2, self.resolution)

        with torch.no_grad():
          pooled_output1 = self.model(**self.processor(images=batch1, return_tensors="pt", do_rescale=False)).last_hidden_state
          pooled_output1 = torch.mean(pooled_output1, dim=1)

          pooled_output2 = self.model(**self.processor(images=batch2, return_tensors="pt", do_rescale=False)).last_hidden_state
          pooled_output2 = torch.mean(pooled_output2, dim=1)

        return torch.mean(sim_matrix(pooled_output1, pooled_output2))