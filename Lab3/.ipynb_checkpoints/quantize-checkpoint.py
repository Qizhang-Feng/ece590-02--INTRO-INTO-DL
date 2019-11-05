import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """

            # Cluster the Weights
            num_centroid = pow(2, bits)

            all_weight = m.conv.weight.data.cpu().detach().numpy()
            weight_shape = all_weight.shape
            
            all_weight = all_weight.reshape(-1,1)
            k_init = np.linspace(all_weight.min(), all_weight.max(), num_centroid)

            kmeans = KMeans(n_clusters=num_centroid, init=k_init.reshape(-1, 1), n_init=1).fit(all_weight)

            # Generate Code Book
            cluster_center = kmeans.cluster_centers_.flatten()

            # Quantize
            indexs = kmeans.predict(all_weight)
            indexs = indexs.reshape(weight_shape)
            
            vmap  = np.vectorize(lambda x:cluster_center[x])
            m.conv.weight.data = torch.from_numpy(vmap(indexs)).to(device)
            
            _cluster_center = [ "{0:b}".format(x).zfill(bits) for x in range(len(cluster_center)) ]
            cluster_centers.append(_cluster_center)

            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
            
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            
            # Cluster the Weights
            num_centroid = pow(2, bits)
            
            all_weight = m.linear.weight.data.cpu().detach().numpy()
            weight_shape = all_weight.shape

            all_weight = all_weight.reshape(-1,1)
            k_init = np.linspace(all_weight.min(), all_weight.max(), num_centroid)
            
            kmeans = KMeans(n_clusters=num_centroid, init=k_init.reshape(-1, 1), n_init=1).fit(all_weight)
            
            # Generate Code Book
            cluster_center = kmeans.cluster_centers_.flatten()

            # Quantize
            indexs = kmeans.predict(all_weight)
            indexs = indexs.reshape(weight_shape)

            vmap  = np.vectorize(lambda x:cluster_center[x])
            m.linear.weight.data = torch.from_numpy(vmap(indexs)).to(device)
            
            _cluster_center = [ "{0:b}".format(x).zfill(bits) for x in range(len(cluster_center)) ]
            cluster_centers.append(_cluster_center)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
            
    return np.array(cluster_centers)

