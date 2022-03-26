import numpy as np
print("import numpy as np")
from ogb.linkproppred import LinkPropPredDataset
print("from ogb.linkproppred import LinkPropPredDataset")


name = 'ogbl-wikikg2'
dataset = LinkPropPredDataset(name)
print("dataset = LinkPropPredDataset(name)")
split_edge = dataset.get_edge_split()
print("split_edge = dataset.get_edge_split()")

train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]
nrelation = int(max(train_triples['relation']))+1

print("type_triples", type(train_triples), type(valid_triples), type(test_triples))

nentity = int(max(np.concatenate((train_triples['head'], 
                                  train_triples['tail']))))+1
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
