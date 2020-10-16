import torch
import torch.nn as nn
import Dutl.embed

class visualizationEmbed(object):
    def __init__(self, embed_network, proj_network, datasets, group_structure):
        self.embed_network = embed_network.eval()
        self.proj_network = proj_network.eval()
        self.datasets = datasets
        self.group_structure = group_structure

    def visualize_group(self, group_index):
        proj_datasets = []
        for data_index in self.group_structure[group_index - 1]:
            embed_data = self.embed_network.embedSingleData(self.datasets[data_index - 1], data_index)
            proj_data = self.proj_network.projectSingleData(embed_data, group_index)
            proj_datasets.append(proj_data)

        return proj_datasets


