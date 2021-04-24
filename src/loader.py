import torch
from torch.utils.data import Dataset, DataLoader


class Tcr_pMhc_Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.weight_factor = 3

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        datum = self.data[idx]
        tcr = datum[2]
        pep = datum[1]
        mhc = datum[0]
        label = self.labels[idx]
        weight = 1
        if label > 0: 
            weight *= self.weight_factor 
        return (tcr, pep, mhc), label
