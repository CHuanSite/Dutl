import Dutl.embed.netStructure as netStructure
from torch.utils.data import DataLoader
import torch.optim as optim

def trainEmbed(datasets,
               group_structure,
               lambda_regularizer = 10,
               epoch_total = 1000,
               batch_size_input = 128):
    p, n_list = netStructure.sizeExtractor(datasets)

    '''
    Embed and Project Network
    '''
    embed_network = netStructure.embedNet(p, n_list)
    proj_network = netStructure.projNet(group_structure)
    embed_network.train()
    proj_network.train()

    '''
    DataLoader
    '''
    dataloader_list = []
    for dataset in datasets:
        dataloader_list.append(DataLoader(dataset, batch_size = batch_size_input, shuffle = True))

    '''
    Training
    '''

    optimizer_embed = optim.Adam(embed_network.parameters())
    optimizer_proj = optim.Adam(proj_network.parameters())

    total_loss = []

    for epoch in range(epoch_total):

        running_loss = 0
        for i in range(1):
            train_datasets = []
            for index, dataset in enumerate(datasets):
                train_datasets.append(next(iter(dataloader_list[index])))

            embed_datasets = embed_network(datasets)
            proj_datasets = proj_network(embed_datasets)

            loss = netStructure.mmdLoss(proj_datasets, group_structure) + lambda_regularizer * proj_network.weightOrthogonal()

            loss.backward()
            optimizer_embed.step()
            optimizer_proj.step()

            running_loss += loss.item()

        total_loss.append(running_loss)
        print(epoch)

    return embed_network, proj_network, total_loss








