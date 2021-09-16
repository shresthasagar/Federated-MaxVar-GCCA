import torch
import matplotlib.pyplot as plt
from dgcca.models import g_step
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import KMeans, SpectralClustering


def cluster_permutation_map(cluster_labels, real_labels, n_labels=10):
    cluster_count = []
    for i in range(n_labels):
        i_count = np.zeros(len(real_labels))
        cluster_temp = cluster_labels.copy() + 1
        
        cluster_temp[cluster_temp!=i+1] = 0
        cluster_temp[cluster_temp==i+1] = 1
        for j in range(n_labels):
            real_temp = (real_labels+1)*cluster_temp
            real_temp[real_temp!=j+1] = 0
            real_temp[real_temp==j+1] = 1
            i_count[j] = real_temp.sum()
        cluster_count.append(i_count)
    return [np.argmax(p) for p in cluster_count]

def map(cluster_labels, cluster_map, n_labels=10):
    cluster_ind = []
    mapped_labels = np.zeros(len(cluster_labels))
    for i in range(n_labels):
        cluster_ind = np.where(cluster_labels==i)
        mapped_labels[cluster_ind] = cluster_map[i]
    return mapped_labels

def delta_sum(mapped_labels, real_labels):
    a = mapped_labels - real_labels
    a[a!=0] = 1
    return len(mapped_labels) - a.sum()

def get_latent_repr(model_path, input_data):
    model = torch.load(model_path)
    out = torch.stack(model(input_data))
    return g_step(out.clone().detach())

def get_clustering_acc(n_clusters=10, gamma=5, random_state=150, latent_repr = None, test_classes=None):
    assert latent_repr is not None
    pred = SpectralClustering(n_clusters=n_clusters, gamma=gamma, random_state=random_state).fit_predict(latent_repr)

    cluster_map = cluster_permutation_map(pred, test_classes.numpy())
    mapped_labels = map(pred, cluster_map)
    acc = delta_sum(mapped_labels, test_classes.numpy())/ len(pred)
    return acc