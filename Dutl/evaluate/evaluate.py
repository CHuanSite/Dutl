import numpy as np
from scipy.spatial import distance_matrix

def probDistance(x, sigma_est = True):
    '''
    Embed data into probabilistic distance matrix
    '''
    x = np.array(x)

    if sigma_est == True:
        sigma = np.mean(np.std(x, 0))
    else:
        sigma = 1

    dist = distance_matrix(x, x)
    exp_dist = np.exp(-1 * dist**2 / sigma**2)
    np.fill_diagonal(exp_dist, 0)
    exp_dist_sum = [[one] for one in np.sum(exp_dist, 1)]
    exp_dist = exp_dist / exp_dist_sum
    return exp_dist

def KL(a, b):
    '''
    Compute KL divergence between two distributions
    '''
    a = np.asarray(a, dtype = np.float)
    b = np.asarray(b, dtype = np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def similarityMeasure(hidden1, hidden2, sigma=10000):
    '''
    Similarity computation from the same component on different datasets
    '''

    ## Normalize the hidden representation
    hidden1 = (hidden1 - np.mean(hidden1, 0)) / np.std(hidden1, 0)
    hidden2 = (hidden2 - np.mean(hidden2, 0)) / np.std(hidden2, 0)

    ## Number of samples
    n1 = hidden1.shape[0]
    n2 = hidden2.shape[0]

    diff_1 = distance_matrix(hidden1, hidden1)
    diff_2 = distance_matrix(hidden2, hidden2)
    diff_12 = distance_matrix(hidden1, hidden2)

    return np.sum(np.exp(-1 * diff_1 ** 2 / sigma)) / (n1 ** 2) + np.sum(np.exp(-1 * diff_2 ** 2 / sigma)) / (
                n2 ** 2) - 2 * np.sum(np.exp(-1 * diff_12 ** 2 / sigma)) / (n1 * n2)

class metricEmbed(object):
    '''
    Class to compute the metrics for the embedding results
    '''

    def __init__(self, datasets, group_structure, embed_network, proj_network, visulization_embed):
        self.datasets = datasets
        self.group_structure = group_structure
        self.embed_network = embed_network
        self.proj_network = proj_network
        self.visulization_embed = visulization_embed


    def geneMetric(self, geneName, group):
        KL_temp = 0
        for index, data_id in enumerate(self.group_structure[group - 1]):
            dist_embed = probDistance(self.visulization_embed.proj_datasets[group - 1][index])
            dist_gene = probDistance(self.datasets[index - 1][geneName])
            KL_temp += KL(dist_embed, dist_gene)

        return KL_temp / len(self.group_structure[group - 1])

    def distributionSimilarity(self, group):
        out_similarity = 0
        N = len(self.visulization_embed.proj_datasets[group - 1])
        count = 0
        for i in range(N - 1):
            for j in range(i, N):
                out_similarity += similarityMeasure(self.visulization_embed.proj_datasets[group - 1][i], self.visulization_embed.proj_datasets[group - 1][j])
                count += 1

        return 1.0 * out_similarity / count

