import torch
import torch.nn as nn
import numpy as np

def sizeExtractor(datasets):
    """
    Extract size of the set of datasets
    """

    p = datasets[0].shape[1]
    n_list = []
    for dataset in datasets:
        n_list.append(dataset.shape[0])

    return p, n_list

class embedNet(nn.Module):
    '''
    Create a embedding network for the datasets
    '''

    def __init__(self, p, n_list):
        super(embedNet, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(p, 256, bias = False),
            nn.ReLU(),
            nn.Linear(256, 64, bias = False),
            nn.ReLU()
        )
        self.normalize = []
        for i in range(n_list - 1):
            self.normalize.append(nn.BatchNorm1d(p))


    def forward(self, datasets):
        embed_datasets = []
        for index, dataset in enumerate(datasets):
            temp_embed = dataset
            if index != 0:
                temp_embed = self.normalize[index - 1](temp_embed)
            for layer in self.embed:
                temp_embed = layer(temp_embed)
            embed_datasets.append(temp_embed)

        return embed_datasets

    '''
    Embed single dataset
    '''
    def c(self, dataset, data_index):
        temp_embed = dataset
        if data_index != 1:
            temp_embed = self.normalize[data_index - 2](temp_embed)
        for layer in self.embed:
            temp_embed = layer(temp_embed)

        return temp_embed




class projNet(nn.Module):

    '''
    Project the embedded layer to low dimension space

    '''

    def __init__(self, group_structure):
        super(projNet, self).__init__()
        self.group_structure = [[group_id - 1 for group_id in group] for group in group_structure]
        self.proj = []
        for i in range(len(group_structure)):
            self.proj.append(nn.Linear(64, 2, bias = False))

    def forward(self, embed_datasets):
        proj_datasets = []
        for index, group in enumerate(self.group_structure):
            proj_datasets.append([])
            for data_id in group:
                proj_datasets[index].append(self.proj[index](embed_datasets[data_id]))

        return proj_datasets

    '''
    Project single dataset based on embed result
    '''

    def projectSingleData(self, embed_data, group_index):
        return self.proj[group_index - 1](embed_data)

    '''
    Condition on the orthogonal of the projection weight
    '''

    def weightOrthogonal(self):
        weight_concatenate = torch.Tensor()
        for layer in self.proj:
            weight_concatenate = torch.cat((weight_concatenate, layer.weight), 1)

        penalty = torch.sum(torch.abs(torch.matmul(torch.transpose(weight_concatenate, 0, 1), weight_concatenate) - torch.eye(weight_concatenate.shape[1])))

        return penalty

def maximumMeanDiscrepancy(embed_data_1,
                           embed_data_2,
                           sigma = 10000):
    '''
    Compute the maximumMeanDiscrepancy between two low dimensional embeddings
    '''

    n1 = embed_data_1.shape[0]
    n2 = embed_data_2.shape[1]

    diff_1 = embed_data_1.unsequeeze(1) - embed_data_1
    diff_2 = embed_data_2.unsequeeze(1) - embed_data_2
    diff_12 = embed_data_1.unsequeeze(1) - embed_data_2

    return torch.sum(torch.exp(-1 * torch.sum(diff_1**2, 2) / sigma)) / (n1**2) + torch.sum(torch.exp(-1 * torch.sum(diff_2**2,2) / sigma)) / (n2**2) - 2 * torch.sum(torch.exp(-1 * torch.sum(diff_12**2, 2) / sigma)) / (n1 * n2)

def mmdLoss(proj_datasets,
            group_structure,
            sigma = 10000):

    out_loss = 0
    for group in group_structure:
        temp_proj_data = []
        temp_loss = 0
        for data_id in group:
            temp_proj_data.append(proj_datasets[data_id])
        for i in range(len(temp_proj_data) - 1):
            for j in range(i, len(temp_proj_data)):
                temp_loss += maximumMeanDiscrepancy(temp_proj_data[i], temp_proj_data[j], sigma = sigma)
        temp_loss = temp_loss / len(temp_proj_data)
        out_loss += temp_loss

    out_loss = out_loss / len(group_structure)

    return out_loss




