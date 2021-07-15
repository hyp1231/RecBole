# @Time   : 2021/7/15
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.dataloader.graph_dataloader
################################################
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from recbole.data.dataloader.general_dataloader import TrainDataLoader, FullSortEvalDataLoader
from recbole.data.interaction import Interaction


def graph_batch_generation(cur_data, graph_objs, item_seq_len):
    index = cur_data['graph_idx']
    graph_batch = {
        k: [graph_objs[k][_.item()] for _ in index]
        for k in graph_objs
    }

    tot_node_num = torch.ones([1], dtype=torch.long)
    for i in range(index.shape[0]):
        for k in graph_batch:
            if 'edge_index' in k:
                graph_batch[k][i] = graph_batch[k][i] + tot_node_num
        graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
        tot_node_num += graph_batch['x'][i].shape[0]

    graph_batch['x'] = [torch.zeros([1], dtype=torch.long)] + graph_batch['x']

    for k in graph_batch:
        if k == 'alias_inputs':
            graph_batch[k] = pad_sequence(graph_batch[k], batch_first=True)
        else:
            graph_batch[k] = torch.cat(graph_batch[k], dim=-1)

    cur_data.update(Interaction(graph_batch))
    return cur_data


class GraphTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        if config['train_neg_sample_args']['strategy'] == 'by':
            raise NotImplementedError('GraphTrainDataLoader doesn\'t support negative sampling currently.')
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        item_seq_len = cur_data[self.config['ITEM_LIST_LENGTH_FIELD']]
        return graph_batch_generation(cur_data, self.dataset.graph_objs, item_seq_len)


class GraphFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        item_seq_len = cur_data[0][self.config['ITEM_LIST_LENGTH_FIELD']]
        converted_graph = graph_batch_generation(cur_data[0], self.dataset.graph_objs, item_seq_len)
        return (converted_graph, *cur_data[1:])
