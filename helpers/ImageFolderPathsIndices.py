import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from os.path import join
from torch.utils.data import DataLoader

# Subclass ImageFolder to return (index, image, label, and file_path)
class ImageFolderPathIndices(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return (index, image, label, path)

def get_dataloader(dataset_path, batch_size, subdir='test', train_aug=True, normalize="ImageNet"):
    
    if normalize == "ImageNet":
        # ImageNet stats
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif normalize == "SC-2000":
        # SC-2000 stats
        mean, std = [0.5762, 0.4830, 0.4713], [0.1401, 0.1541, 0.1582]
    elif normalize == "SC-2000v2":
        # SC-2000v2 stats
        mean, std = [0.5744, 0.4826, 0.4668], [0.1406, 0.1537, 0.1575]
    elif normalize == "none":
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    else:
        raise ValueError("Must choose from valid dataset name: [\'ImageNet\', \'SC-2000\']")
    

    # Includes random transformations for on-the-fly augmentation
    random_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])

    # Only includes deterministic transforms
    stable_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])
    
    # Determine which folder in dataset to open
    if subdir == 'train':
        # Set whether or not to use on-the-fly augmentation during training
        if train_aug:
            dataset = ImageFolderPathIndices(join(dataset_path, 'train'), transform=random_transforms)
        else:
            dataset = ImageFolderPathIndices(join(dataset_path, 'train'), transform=stable_transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    elif subdir == 'valid':
        dataset = ImageFolderPathIndices(join(dataset_path, 'valid'), transform=stable_transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif subdir == 'test':
        dataset = ImageFolderPathIndices(join(dataset_path, 'test'), transform=stable_transforms)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        raise ValueError('Must set subdir to one of [\'train\', \'valid\', \'test\'].')

def get_mean_std(dataloader):
    dataset_size = len(dataloader.dataset)
    running_mean = torch.zeros(3, dtype=torch.float32)
    running_std = torch.zeros(3, dtype=torch.float32)
    for _, imgs, _, _ in dataloader:
        # Take the mean along height and width axes of image batch
        # Sum the means along batch axis
        # Add to running total of means
        running_mean += torch.sum(imgs.mean([2,3]), axis=0)
        running_std += torch.sum(imgs.std([2,3]), axis=0)
    
    # Return mean and std sums divided by dataset size
    return running_mean / dataset_size, running_std / dataset_size