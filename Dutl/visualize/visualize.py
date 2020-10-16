import matplotlib.pyplot as plt

class visualizationEmbed(object):
    def __init__(self,
                 embed_network,
                 proj_network,
                 datasets,
                 explains,
                 group_structure):

        self.embed_network = embed_network.eval()
        self.proj_network = proj_network.eval()
        self.datasets = datasets
        self.explains = explains
        self.group_structure = group_structure

        self.proj_datasets = self.visualize_all()

    def visualize_group(self, group_index):
        proj_datasets = []
        for data_index in self.group_structure[group_index - 1]:
            embed_data = self.embed_network.embedSingleData(self.datasets[data_index - 1], data_index)
            proj_data = self.proj_network.projectSingleData(embed_data, group_index)
            proj_datasets.append(proj_data)

        return proj_datasets

    def visualize_all(self):
        proj_datasets = []
        for i in range(len(self.group_structure)):
            proj_datasets.append(self.visualize_group(i + 1))

        return proj_datasets

    def showImage(self, group_index):
        for data_index in self.group_structure[group_index - 1]:
            embed_data = self.embed_network.embedSingleData(self.datasets[data_index - 1], data_index)
            proj_data = self.proj_network.projectSingleData(embed_data, group_index)
            plt.scatter(proj_data.detach()[:, 0], proj_data.detach()[:, 1], c = self.explains[data_index - 1]['COL'])
            plt.show()




