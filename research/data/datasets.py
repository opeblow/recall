"""Dataset utilities for research project."""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, CIFAR10


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.247, 0.243, 0.261]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_split_cifar10(task_id: int, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 split by task_id.
    
    Args:
        task_id: Task ID (0-4), each task has 2 classes.
        batch_size: Batch size for DataLoader.
        
    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    train_full = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    test_full = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
    
    start_class = task_id * 2
    end_class = start_class + 2
    classes = list(range(start_class, end_class))
    
    train_indices = [i for i, label in enumerate(train_full.targets) if label in classes]
    test_indices = [i for i, label in enumerate(test_full.targets) if label in classes]
    
    train_subset = SubsetDataset(train_full, train_indices)
    test_subset = SubsetDataset(test_full, test_indices)
    
    remap_labels = {old: new for new, old in enumerate(classes)}
    
    class RemappedDataset(Dataset):
        def __init__(self, subset, remap):
            self.subset = subset
            self.remap = remap
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return img, self.remap[label]
    
    train_dataset = RemappedDataset(train_subset, remap_labels)
    test_dataset = RemappedDataset(test_subset, remap_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def get_permuted_mnist(task_id: int, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST with task-specific pixel permutation.
    
    Args:
        task_id: Task ID for random seed and permutation.
        batch_size: Batch size for DataLoader.
        
    Returns:
        Tuple of (train_loader, test_loader).
    """
    torch.manual_seed(task_id)
    
    num_pixels = 28 * 28
    permutation = torch.randperm(num_pixels)
    
    class PermutedMNIST(Dataset):
        def __init__(self, dataset: Dataset, permutation: torch.Tensor):
            self.dataset = dataset
            self.permutation = permutation
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            img = img.view(-1)
            img = img[self.permutation]
            img = img.view(1, 28, 28)
            return img, label
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_full = MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    test_full = MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    
    train_dataset = PermutedMNIST(train_full, permutation)
    test_dataset = PermutedMNIST(test_full, permutation)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def get_split_cub200(task_id: int, data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Load CUB-200-2011 split by task_id.
    
    Args:
        task_id: Task ID (0-9), each task has 20 classes.
        data_dir: Directory containing CUB-200 data.
        batch_size: Batch size for DataLoader.
        
    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("CUB-200-2011 dataset not found.")
        print("\n" + "=" * 60)
        print("DOWNLOAD INSTRUCTIONS FOR CUB-200-2011")
        print("=" * 60)
        print("1. Download the dataset from:")
        print("   https://www.vision.caltech.edu/datasets/cub2002011/")
        print("\n2. Extract the archive. The expected structure is:")
        print(f"   {data_dir}/")
        print("       ├── images/")
        print("       ├── images.txt")
        print("       ├── train_test_split.txt")
        print("       └── ... (other annotation files)")
        print("\n3. Alternatively, use the following commands:")
        print("   wget https://www.vision.caltech.edu/datasets/cub2002011/CUB_200_2011.tgz")
        print("   tar -xzvf CUB_200_2011.tgz")
        print("   mv CUB_200_2011 <your_data_dir>")
        print("=" * 60)
        raise FileNotFoundError(f"CUB-200-2011 dataset not found at {data_dir}")
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    train_full = datasets.ImageFolder(train_dir, transform=transform_train)
    test_full = datasets.ImageFolder(test_dir, transform=transform_test)
    
    num_classes_per_task = 20
    start_class = task_id * num_classes_per_task
    end_class = start_class + num_classes_per_task
    
    train_indices = [i for i, (_, label) in enumerate(train_full.samples) if start_class <= label < end_class]
    test_indices = [i for i, (_, label) in enumerate(test_full.samples) if start_class <= label < end_class]
    
    remap_labels = {old: new for new, old in enumerate(range(start_class, end_class))}
    
    class RemappedCubDataset(Dataset):
        def __init__(self, subset, remap):
            self.subset = subset
            self.remap = remap
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return img, self.remap[label]
    
    train_subset = Subset(train_full, train_indices)
    test_subset = Subset(test_full, test_indices)
    
    train_dataset = RemappedCubDataset(train_subset, remap_labels)
    test_dataset = RemappedCubDataset(test_subset, remap_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


class SimpleCNN(nn.Module):
    """Simple CNN backbone for CIFAR-10 and MNIST."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        if input_channels == 3:
            fc_input_size = 128 * 4 * 4
        else:
            fc_input_size = 128 * 3 * 3
        
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_head(x)
        return x


if __name__ == "__main__":
    print("Testing get_split_cifar10...")
    train_loader, test_loader = get_split_cifar10(task_id=0, batch_size=64)
    for images, labels in train_loader:
        print(f"  CIFAR10 train batch: images={images.shape}, labels={labels.shape}")
        break
    for images, labels in test_loader:
        print(f"  CIFAR10 test batch: images={images.shape}, labels={labels.shape}")
        break
    
    print("\nTesting get_permuted_mnist...")
    train_loader, test_loader = get_permuted_mnist(task_id=0, batch_size=64)
    for images, labels in train_loader:
        print(f"  MNIST train batch: images={images.shape}, labels={labels.shape}")
        break
    for images, labels in test_loader:
        print(f"  MNIST test batch: images={images.shape}, labels={labels.shape}")
        break
    
    print("\nTesting get_split_cub200...")
    try:
        train_loader, test_loader = get_split_cub200(task_id=0, data_dir='./data/cub200', batch_size=32)
        for images, labels in train_loader:
            print(f"  CUB200 train batch: images={images.shape}, labels={labels.shape}")
            break
        for images, labels in test_loader:
            print(f"  CUB200 test batch: images={images.shape}, labels={labels.shape}")
            break
    except FileNotFoundError as e:
        print(f"  Expected error: {e}")
    
    print("\nTesting SimpleCNN...")
    model_cifar = SimpleCNN(num_classes=2, input_channels=3)
    x_cifar = torch.randn(4, 3, 32, 32)
    out_cifar = model_cifar(x_cifar)
    print(f"  CIFAR input: {x_cifar.shape} -> output: {out_cifar.shape}")
    
    model_mnist = SimpleCNN(num_classes=10, input_channels=1)
    x_mnist = torch.randn(4, 1, 28, 28)
    out_mnist = model_mnist(x_mnist)
    print(f"  MNIST input: {x_mnist.shape} -> output: {out_mnist.shape}")
    
    print("\nAll tests passed!")
