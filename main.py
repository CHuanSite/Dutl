import torch

import Dutl.visualize.visualize as visualize
import Dutl.embed.train as train
import Dutl.evaluate.evaluate as evaluate

import numpy as np
import pandas as pd

#%%

'''
Read data into Python
'''

data_file_name = ['/Users/huanchen/Documents/autoencoder_Genomics/data/1_inVitro_Bulk_Cortecon.plog2_trimNord.txt',
                  '/Users/huanchen/Documents/autoencoder_Genomics/data/2_inVitro_SingleCell_scESCdifBifurc.CelSeq_trimNord.txt',
                  '/Users/huanchen/Documents/autoencoder_Genomics/data/3_inVivo_Bulk_BrainSpan_RNAseq_Gene_DFC_noSVA_plog2_trimNord.txt',
                  '/Users/huanchen/Documents/autoencoder_Genomics/data/4_inVivo_SingleCell_CtxDevoSC4kTopoTypoTempo_plog2_trimNord.txt']

datasets = []

for one in data_file_name:
    dat_temp = pd.read_table(one)
    dat_temp = dat_temp.drop(dat_temp.columns[0], axis = 1)
    dat_temp = dat_temp.sub(dat_temp.mean(1), axis = 0)
    dat_temp = dat_temp.sub(dat_temp.mean(0), axis = 1)
    dat_temp = dat_temp.divide(np.sqrt(dat_temp.var(0)), axis = 1)
    datasets.append(torch.transpose(torch.tensor(dat_temp.values.astype(np.float32)), 0, 1))

#%%

'''
Read explaination into Python
'''

explain_file_name = ['/Users/huanchen/Documents/autoencoder_Genomics/data_explanation/1_inVitro_Bulk_Cortecon.pd.txt',
                     '/Users/huanchen/Documents/autoencoder_Genomics/data_explanation/2_inVitro_SingleCell_scESCdifBifurc.CelSeq.pd.txt',
                     '/Users/huanchen/Documents/autoencoder_Genomics/data_explanation/3_inVivo_Bulk_BrainSpan.RNAseq.Gene.DFC.pd.txt',
                     '/Users/huanchen/Documents/autoencoder_Genomics/data_explanation/4_inVivo_SingleCell_CtxDevoSC4kTopoTypoTempo.pd.txt']

explains = []
for index, one in enumerate(explain_file_name):
    dat_temp = pd.read_table(one)
    if index == 0 or index == 2:
        dat_temp['COL'] = dat_temp['color']
    else:
        dat_temp['COL'] = dat_temp['COLORby.DCX']
    explains.append(dat_temp)

#%%

'''
Train neural network
'''
group_structure = [[1,2,3,4], [1,2], [3,4], [1,3], [2,4]]
embed_network, proj_network, total_loss = train.trainEmbed(datasets, group_structure, lambda_regularizer = 10, epoch_total = 100, batch_size_input = 128)


#%%

'''
Visualize datasets
'''

visualization_obj = visualize.visualizationEmbed(embed_network, proj_network, datasets, explains, group_structure)

visualization_obj.showImage(1)
visualization_obj.showImage(2)
visualization_obj.showImage(3)
visualization_obj.showImage(4)
visualization_obj.showImage(5)


#%%

'''
Metric of the embedding
'''

metricEmbed_obj = evaluate.metricEmbed(datasets, group_structure, embed_network, proj_network, visualization_obj)