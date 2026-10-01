"""
HOG planar graph dataset for DiGress.
Inherits from AbstractDataModule and AbstractDatasetInfos for proper integration.
"""

import os
import pathlib
import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.analysis.spectre_utils import PlanarSamplingMetrics
import src.utils as utils


class HOGPlanarDataset(torch_geometric.data.InMemoryDataset):
    """Load HOG planar graphs from pt file."""
    
    def __init__(self, file_path: str, split: str = 'train', transform=None, pre_transform=None, pre_filter=None):
        self.file_path = file_path
        self.split = split
        self.num_graphs = 4000
        self.processed_version = 'v2'
        super().__init__(root=os.path.dirname(file_path), transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    @property
    def processed_file_names(self):
        return [f'{self.split}_{self.processed_version}.pt']
    
    def download(self):
        """Load and split raw graphs."""
        raw_graphs = torch.load(self.file_path, weights_only=False)
        
        # Parse raw data
        data_list = []
        graphs = raw_graphs if isinstance(raw_graphs, list) else [raw_graphs]

        for sample in graphs:
            if isinstance(sample, torch_geometric.data.Data):
                if getattr(sample, 'edge_attr', None) is not None and sample.edge_attr.dim() == 2 and sample.edge_attr.size(1) >= 2:
                    data_list.append(sample)
                    continue
                n = sample.x.size(0) if getattr(sample, 'x', None) is not None else int(sample.n_nodes.item())
                adj = torch.zeros((n, n), dtype=torch.float)
                adj[sample.edge_index[0], sample.edge_index[1]] = 1.0
            else:
                adj = sample

            if len(adj.shape) != 2:
                continue

            n = adj.shape[0]
            x = torch.ones(n, 1, dtype=torch.float)
            edge_index, _ = dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            y = torch.zeros([1, 0]).float()
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                n_nodes=num_nodes
            )
            data_list.append(data)
        
        # Split: 64% train, 16% val, 20% test
        n_total = len(data_list)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        
        test_len = int(round(n_total * 0.2))
        train_len = int(round((n_total - test_len) * 0.8))
        val_len = n_total - train_len - test_len
        
        indices = torch.randperm(n_total, generator=g_cpu)
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]
        
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])
    
    def process(self):
        """Process split data."""
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_list = torch.load(self.raw_paths[file_idx[self.split]], weights_only=False)
        
        processed_data_list = []
        for sample in raw_list:
            if isinstance(sample, torch_geometric.data.Data):
                n = sample.x.size(0) if getattr(sample, 'x', None) is not None else int(sample.n_nodes.item())
                if getattr(sample, 'edge_attr', None) is not None and sample.edge_attr.dim() == 2 and sample.edge_attr.size(1) == 2:
                    data = sample
                else:
                    edge_index = sample.edge_index
                    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                    edge_attr[:, 1] = 1
                    x = sample.x if getattr(sample, 'x', None) is not None else torch.ones(n, 1, dtype=torch.float)
                    y = sample.y if getattr(sample, 'y', None) is not None else torch.zeros([1, 0]).float()
                    num_nodes = sample.n_nodes if getattr(sample, 'n_nodes', None) is not None else n * torch.ones(1, dtype=torch.long)
                    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                                     y=y, n_nodes=num_nodes)
            else:
                adj = sample
                n = adj.shape[0]
                x = torch.ones(n, 1, dtype=torch.float)
                y = torch.zeros([1, 0]).float()
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                num_nodes = n * torch.ones(1, dtype=torch.long)
                data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                                 y=y, n_nodes=num_nodes)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            processed_data_list.append(data)
        
        torch.save(self.collate(processed_data_list), self.processed_paths[0])


class HOGPlanarDataModule(AbstractDataModule):
    """DataModule for HOG planar graphs, inherits from AbstractDataModule."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[5]
        self.hog_file_path = os.path.join(base_path, 'ext', 'hog_planar', 'planar_graph.pt')
        
        datasets = {
            'train': HOGPlanarDataset(file_path=self.hog_file_path, split='train'),
            'val': HOGPlanarDataset(file_path=self.hog_file_path, split='val'),
            'test': HOGPlanarDataset(file_path=self.hog_file_path, split='test')
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset
    
    def __getitem__(self, item):
        return self.inner[item]


class HOGDatasetInfos(AbstractDatasetInfos):
    """Dataset info for HOG planar graphs, inherits from AbstractDatasetInfos."""
    
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'hog_planar'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1.0])  # Single node type
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
    
    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        """Compute dimensions from example batch."""
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        
        # Handle y=None from batching (unconditional datasets with empty y)
        if example_batch.y is None:
            y_t = torch.zeros(example_batch.num_graphs, 0, dtype=torch.float)
        else:
            y_t = example_batch.y
        
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': y_t, 'node_mask': node_mask}

        # Add 1 for time conditioning (will be concatenated with y in compute_extra_data)
        self.input_dims = {'X': example_batch.x.size(1),
                           'E': example_batch.edge_attr.size(1),
                           'y': y_t.size(1) + 1}
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch.x.size(1),
                            'E': example_batch.edge_attr.size(1),
                            'y': 0}


# Use SPECTRE's comprehensive PlanarSamplingMetrics instead of custom implementation
# PlanarSamplingMetrics computes: degree, clustering, orbit, spectre, planarity, uniqueness, and non-isomorphism stats
DummySamplingMetrics = PlanarSamplingMetrics

