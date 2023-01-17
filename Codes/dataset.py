import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from .augmentations import crop
from .utils import load_graph_txt

class SynthDataset(Dataset):
    
    def __init__(self, train=True, cropSize=(96,96,96), th=15):
        
        image_path = {"train": ["/cvlabdata2/home/oner/Snakes/codes/Synth/images/data_{}.npy".format(i) for i in range(20)],
                      "val":  ["/cvlabdata2/home/oner/Snakes/codes/Synth/images/data_{}.npy".format(i) for i in range(20,30)]}
        label_path = {"train": ["/cvlabdata2/home/oner/Snakes/codes/Synth/dist_labels/data_{}.npy".format(i) for i in range(20)],
                      "val":  ["/cvlabdata2/home/oner/Snakes/codes/Synth/dist_labels/data_{}.npy".format(i) for i in range(20,30)]}
        graph_path = {"train": ["/cvlabdata2/home/oner/Snakes/codes/Synth/graphs_inv/data_{}.graph".format(i) for i in range(20)],
                      "val":  ["/cvlabdata2/home/oner/Snakes/codes/Synth/graphs_inv/data_{}.graph".format(i) for i in range(20,30)]}
        
        self.images = image_path["train"] if train else image_path["val"]
        self.labels = label_path["train"] if train else label_path["val"]
        self.graphs = graph_path["train"] if train else graph_path["val"]
            
        self.train = train
        self.cropSize = cropSize
        self.th = th
        
    def __getitem__(self, index):

        image = np.load(self.images[index])
        label = np.load(self.labels[index])
        graph = load_graph_txt(self.graphs[index])
        
#         for n in graph.nodes:
#             graph.nodes[n]["pos"]=graph.nodes[n]["pos"][-1::-1]
            
        slices = None
        
        if self.train:
            image, label, slices = crop([image, label], self.cropSize)
            
        label[label>self.th] = self.th
        
        if self.train:
            return torch.tensor(image), torch.tensor(label), graph, slices
        
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.images)

def collate_fn(data):
    transposed_data = list(zip(*data))
    images = torch.stack(transposed_data[0], 0)[:,None]
    labels = torch.stack(transposed_data[1], 0)[:,None]
    graphs = transposed_data[2]
    slices = transposed_data[3]
    
    return images, labels, graphs, slices