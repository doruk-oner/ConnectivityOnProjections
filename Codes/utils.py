import os
import yaml
import torch
import logging
import pickle
import networkx as nx
from . import cropping

torch_version_major = int(torch.__version__.split('.')[0])
torch_version_minor = int(torch.__version__.split('.')[1])

class Dummysink(object):
    def write(self, data):
        pass # ignore the data
    def __enter__(self): return self
    def __exit__(*x): pass

torch_no_grad = Dummysink() if torch_version_major==0 and torch_version_minor<4 else torch.no_grad()

def to_torch(ndarray, volatile=False):
    if torch_version_major>=1:
        return torch.from_numpy(ndarray)
    else:
        from torch.autograd import Variable
        return Variable(torch.from_numpy(ndarray), volatile=volatile)

def from_torch(tensor, num=False):
    return tensor.data.cpu().numpy()

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def yaml_read(filename):
    try:
        with open(filename, 'r') as f:
            try:
                data = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError as e:
                data = yaml.load(f)
        return data
    except:
        raise ValueError("Unable to read YAML {}".format(filename))

def pickle_read(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def torch_save(filename, dictionary):
    mkdir(os.path.dirname(filename))
    torch.save(dictionary, filename)

def config_logger(log_file=None):
    """
    Basic configuration of the logging system. Support logging to a file.
    Log messages can be submitted from any script.
    config_logger(.) is called once from the main script.

    Example
    -------
    import logging
    logger = logging.getLogger(__name__)
    utils.config_logger("main.log")
    logger.info("this is a log.")
    """

    class MyFormatter(logging.Formatter):

        info_format = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        error_format = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"

        def format(self, record):

            if record.levelno > logging.INFO:
                self._style._fmt = self.error_format
            else:
                self._style._fmt = self.info_format

            return super(MyFormatter, self).format(record)

    rootLogger = logging.getLogger()

    if log_file is not None:
        fileHandler = logging.FileHandler(log_file)
        fileFormatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s")
        fileHandler.setFormatter(fileFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleFormatter = MyFormatter()
    consoleHandler.setFormatter(consoleFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

def process_in_chuncks(image, output, process, patch_size, patch_margin):
    """
    N,C,D1,D2,...,Dn
    """
    assert len(image.shape)==len(output.shape)
    assert (len(image.shape)-2)==len(patch_size)
    assert len(patch_margin)==len(patch_size)

    chunck_coords = cropping.split_with_margin(image.shape[2:], patch_size, patch_margin)

    semicol = (slice(None,None),) # this mimicks :

    for source_c, valid_c, destin_c in zip(*chunck_coords):

        crop = image[semicol+semicol+source_c]
        proc_crop = process(crop)
        output[semicol+semicol+destin_c] = proc_crop[semicol+semicol+valid_c]

    return output

def load_graph_txt(filename):
     
    G = nx.Graph()
        
    nodes = []
    edges = []
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 and switch:
                switch = False
                continue
            if switch:
                x,y,z = line.split(' ')
                G.add_node(i, pos=(float(x),float(y),float(z)))
                i+=1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1),int(idx_node2))
    
    return G

def save_graph_txt(G, filename):
    
    mkdir(os.path.dirname(filename))
    
    nodes = list(G.nodes())
    
    file = open(filename, "w+")
    for n in nodes:
        file.write("{:.6f} {:.6f} {:.6f}\r\n".format(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1], G.nodes[n]['pos'][2]))
    file.write("\r\n")
    for s,t in G.edges():
        file.write("{} {}\r\n".format(nodes.index(s), nodes.index(t)))
    file.close()
    
def interpolate_new_nodes(p1, p2, spacing=2):

    p1_, p2_ = np.array(p1), np.array(p2)

    segment_length = np.linalg.norm(p1_-p2_)

    new_node_pos = p1_ + (p2_-p1_)*np.linspace(0,1,int(np.ceil(segment_length/spacing)))[1:-1,None]

    return new_node_pos

def oversampling_graph(G, spacing=20):
    edges = list(G.edges())
    for s,t in edges:

        new_nodes_pos = interpolate_new_nodes(G.nodes[s]['pos'], G.nodes[t]['pos'], spacing)

        if len(new_nodes_pos)>0:
            G.remove_edge(s,t)
            n = max(G.nodes())+1

            for i,n_pos in enumerate(new_nodes_pos):
                G.add_node(n+i, pos=tuple(n_pos))

            G.add_edge(s,n)
            for _ in range(len(new_nodes_pos)-1):
                G.add_edge(n,n+1)
                n+=1
            G.add_edge(n,t)
    return G