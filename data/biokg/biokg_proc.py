import numpy as np
print("import numpy as np")
from ogb.linkproppred import LinkPropPredDataset
print("from ogb.linkproppred import LinkPropPredDataset")


name = 'ogbl-biokg'
dataset = LinkPropPredDataset(name)
print("dataset = LinkPropPredDataset(name)")
split_edge = dataset.get_edge_split()
print("split_edge = dataset.get_edge_split()")

train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

cur_idx, cur_type_idx, type_dict, entity_dict = 0, 0, {}, {}
for key in dataset[0]['num_nodes_dict']:
    type_dict[key] = cur_type_idx
    cur_type_idx += 1
    entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
    cur_idx += dataset[0]['num_nodes_dict'][key]
    print("key", key)
print("cur_idx, cur_type_idx, type_dict, entity_dict", cur_idx, cur_type_idx, type_dict, entity_dict)

def index_triples_across_type(triples, entity_dict, type_dict):
    triples['head_type_idx'] = np.zeros_like(triples['head'])
    triples['tail_type_idx'] = np.zeros_like(triples['tail'])
    for i in range(len(triples['head'])):
        h_type = triples['head_type'][i]
        triples['head_type_idx'][i] = type_dict[h_type] 
        triples['head'][i] += entity_dict[h_type][0]
        if 'head_neg' in triples:
            triples['head_neg'][i] += entity_dict[h_type][0]
        t_type = triples['tail_type'][i]
        triples['tail_type_idx'][i] = type_dict[t_type]
        triples['tail'][i] += entity_dict[t_type][0]
        if 'tail_neg' in triples:
            triples['tail_neg'][i] += entity_dict[t_type][0]
    return triples

print('Indexing triples across different entity types ...')
train_triples = index_triples_across_type(train_triples, entity_dict, type_dict)
valid_triples = index_triples_across_type(valid_triples, entity_dict, type_dict)
test_triples = index_triples_across_type(test_triples, entity_dict, type_dict)
nrelation = int(max(train_triples['relation']))+1
nentity = sum(dataset[0]['num_nodes_dict'].values())
assert train_triples['head'].max() <= nentity

print(nentity, nrelation)

train_array = np.concatenate((train_triples['head'].reshape(-1, 1),
                              train_triples['relation'].reshape(-1, 1),
                              train_triples['tail'].reshape(-1, 1),
                              ), axis=1)

valid_array = np.concatenate((valid_triples['head'].reshape(-1, 1),
                              valid_triples['relation'].reshape(-1, 1),
                              valid_triples['tail'].reshape(-1, 1),
                              valid_triples['head_neg'],
                              valid_triples['tail_neg'],
                              ), axis=1)

test_array = np.concatenate((test_triples['head'].reshape(-1, 1),
                              test_triples['relation'].reshape(-1, 1),
                              test_triples['tail'].reshape(-1, 1),
                              test_triples['head_neg'],
                              test_triples['tail_neg'],
                              ), axis=1)

print("shape: ", train_array.shape, valid_array.shape, test_array.shape)

array = [train_array, valid_array, test_array]
f_names = ['train', 'valid', 'test']
for i in range(3):
    f = open(f_names[i]+".del", 'w')
    for j in range(len(array[i])):
        for k in range(len(array[i][j])):
            f.write(str(array[i][j][k]))
            if k<len(array[i][j])-1: 
                f.write('\t')
        f.write('\n')
    f.close()
