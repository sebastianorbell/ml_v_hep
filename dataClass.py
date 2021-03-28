import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
torchType = torch.float32

class jetDataset(Dataset):
    """jet images dataset."""

    def __init__(self, npyFileClass1, npyFileClass2, root_dir, transform=None, torchType=torch.float32):
        """
        Args:
            npyFileClass1 (string): Path to the npy file of class 1.
            npyFileClass2 (string): Path to the npy file of class 2.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.torchType = torchType

        self.class1 = np.load(npyFileClass1)
        self.labels1 = np.zeros([np.shape(self.class1)[0]],dtype=int)
        self.class2 = np.load(npyFileClass2)
        self.labels2 = np.ones([np.shape(self.class2)[0]],dtype=int)

        self.allData = []
        self.allLabels = []
        self.allData.extend(self.class1)
        self.allLabels.extend(self.labels1)
        self.allData.extend(self.class2)
        self.allLabels.extend(self.labels2)

        self.allData = np.array(self.allData)
        self.allData = np.log(self.allData, out=np.zeros_like(self.allData), where=(self.allData != 0))
        self.allLabels = np.array(self.allLabels)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.allLabels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.allData[idx]
        label = self.allLabels[idx]

        if self.transform:
            image = self.transform(image)
            image = image.type(self.torchType)

        sample = {'image': image, 'label': label}

        return sample


def imshow(imgs, labels=None):
    imgs = imgs
    try:
        npimgs = imgs.numpy()
    except:
        npimgs = imgs
    n = len(npimgs)
    fig, ax = plt.subplots(ncols=n)
    for i in range(n):
        ax[i].imshow(np.abs(npimgs[i]), cmap="jet")#, norm=LogNorm())
        if isinstance(labels[i], str):
            ax[i].set_title(labels[i])
    plt.show()