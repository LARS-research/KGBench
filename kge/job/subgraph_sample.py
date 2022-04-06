import os
from kge.job import AutoSearchJob, Job
from kge import Config
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from torch.optim import SGD
import networkx as nx
import pickle as pkl  
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import yaml
import math

class SubgraphSample(AutoSearchJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.sample_ratio = self.config.get("subgraph_sample.sampled_ratio")
        self.num_trials = self.config.get("subgraph_sample.num_trials")
        self.num_starts = self.config.get("subgraph_sample.num_starts")
        self.repeat = self.config.get("subgraph_sample.repeat")
        self.dataset_name = self.config.get("dataset.name")
        self.parameters_config = self.config.get("subgraph_sample.parameters")

        # self.structure_genarator_mode = self.config.get("sf_search.structure_genarator_mode")
        # self.prob = self.config.get("sf_search.prob")
        # self.K = self.config.get("sf_search.K")

        if self.__class__ == SubgraphSample:
            for f in Job.job_created_hooks:
                f(self)

        args = { }
        args['sample_ratio'] = self.sample_ratio
        args['dataset'] = self.dataset_name
        args['addInverseRelation'] = False

        print('preparing training data ...')
        pklFile = os.path.join('./data', self.dataset_name, 'datasetInfo.pkl')
        if os.path.exists(pklFile):
            datasetInfo = pkl.load(open(pklFile, 'rb')) 
        else:
            args['addInverseRelation'] = False
            datasetInfo = prepareData(args)
            pkl.dump(datasetInfo, open(pklFile, "wb"), protocol=4)
        print('==> finish loading dataset')

        nentity       = datasetInfo['nentity']
        nrelation     = datasetInfo['nrelation']
        # sample_ratio  = args.sample_ratio
        # num_starts    = args.num_starts
        savePath      = "./data"
        ori_train_triples = datasetInfo['train_triples']
        train_triples = []

        
        if args['dataset'] in ['biokg', 'wikikg2', 'ogbl-biokg', 'ogbl-wikikg2']:
            for idx in tqdm(range(len(ori_train_triples['head']))):
                head, relation, tail = ori_train_triples['head'][idx], ori_train_triples['relation'][idx], ori_train_triples['tail'][idx]
                if relation < nrelation:
                    train_triples.append((head, relation, tail))
        else:
            for head, relation, tail in tqdm(ori_train_triples):
                if relation < nrelation:
                    train_triples.append((head, relation, tail))

        if self.repeat == 0:
            self.sampled_dataset = KGGenerator(self.dataset_name, train_triples, self.sample_ratio, self.num_starts, savePath).sampled_dataset
        else:
            for repeat_num in range(1, self.repeat+1):
               self.sampled_dataset = KGGenerator(self.dataset_name, train_triples, self.sample_ratio, self.num_starts, savePath, repeat_num=repeat_num).sampled_dataset 

        self.trial_id = -1
        self.exp_config = {}
        self.record_sampled = []
        self.record_whole = []
        
    def register_trial(self, parameters=None):
        # parameters: 优化参数的字典
        # trail_id: 训练次数, int
        return {}, 0


    def register_trial_result(self, trial_id, parameters, trace_entry):
        # trace_entry["metric_value"]: MRR, float
        # parameters: 优化参数的字典
        pass


    def get_best_parameters(self):
        # 仅仅是继承了这个函数名，实际功能为输出实验结果
        # print("mrr on sampled graph: ", self.record_sampled)
        # print("mrr on whole graph: ", self.record_whole)
        # num_exps = len(record_s)
        # # record_s = np.array(self.record_sampled)
        # # record_s_arg = np.argsort(record_s)
        # # sampled_ranking = np.zeros(num_exps)
        # # for idx in range(num_exps):
        # #     sampled_ranking[record_s_arg[idx]] = idx
        # # record_w = np.array(self.record_whole)
        # # record_w_arg = np.argsort(record_w)
        # # whole_ranking = np.zeros(num_exps)
        # # for idx in range(num_exps):
        # #     whole_ranking[record_w_arg[idx]] = idx
        # # print("config rank on sampled graph: ", sampled_ranking)
        # # print("config rank on whole graph: ", whole_ranking)
        # # srcc = 1 - abs(sampled_ranking - whole_ranking).sum()/(num_exps*(num_exps**2-1))
        # df = pd.DataFrame({'whole': self.record_whole, 'sampled': self.record_sampled})
        # print("srcc: ", df.corr('spearman'))
        # self.config.log(
        #                 "mrr on sampled graph: {}\n mrr on whole graph:{}\n srcc:{}".format(
        #                     self.record_sampled, self.record_whole, srcc
        #                 )
        #             )
        pass

    # def ramdom_config_genarator(self):
    #     new_config = {}
    #     for idx in range(len(self.parameters_config)):
    #         bound = self.parameters_config[idx]['bounds'] 
    #         if self.parameters_config[idx]['type'] == 'log':
    #             new_config[self.parameters_config[idx]['name']] = math.exp(random.uniform(math.log(bound[0]), math.log(bound[1])))
    #         elif self.parameters_config[idx]['type'] == 'range':
    #             new_config[self.parameters_config[idx]['name']] = random.uniform(bound[0], bound[1])
    #         elif self.parameters_config[idx]['type'] == 'choice':
    #             new_config[self.parameters_config[idx]['name']] = bound[random.randint(0, len(bound)-1)]
    #         else:
    #             new_config[self.parameters_config[idx]['name']] = bound[0]
    #     return new_config
    


class KGGenerator:
    def __init__(self, dataset, all_triples, sample_ratio, num_starts, savePath, repeat_num=0, split_ratio=[0.9, 0.05, 0.05]):        
        self.split_ratio  = split_ratio
        self.sample_ratio = sample_ratio
        self.repeat_num   = repeat_num
        self.num_starts   = num_starts
        self.dataset      = dataset

        # setup graph
        homoGraph = self.triplesToNxGraph(all_triples)
        diGraph   = self.triplesToNxDiGraph(all_triples)

        # sampling via random walk
        num_nodes = homoGraph.number_of_nodes()
        target_num_nodes = int(num_nodes * self.sample_ratio)
        if num_starts == 1:
            sampled_nodes = self.random_walk_induced_graph_sampling(homoGraph, target_num_nodes)
        else:
            # sampled_nodes = []
            # for i in range(int(num_starts)):
            #     sampled_nodes += self.random_walk_induced_graph_sampling(homoGraph, int(1.2 * target_num_nodes / num_starts))
            # sampled_nodes = list(set(sampled_nodes))[:target_num_nodes]
            sampled_nodes = self.multi_starts_random_walk_induced_graph_sampling(homoGraph, target_num_nodes, num_starts)

        sampled_graph      = diGraph.subgraph(sampled_nodes)
        homo_sampled_graph = homoGraph.subgraph(sampled_nodes)

        # show num of connected parts
        connected_components = nx.connected_components(homo_sampled_graph)
        connected_subgraphs = []
        for c in connected_components:
            connected_subgraphs.append(homo_sampled_graph.subgraph(c).copy())
        print(f'==> #connected_components:{len(connected_subgraphs)}')

        # build sampled KG
        self.all_triples = []
        self.relations   = []
        self.entities    = []
        for edge in list(sampled_graph.edges(data=True)):
            h,t = edge[0], edge[1]
            r = edge[2]['relation']
            self.all_triples.append((h,r,t))
            self.relations.append(r)
            self.entities.append(h)
            self.entities.append(t)

        # assign new index to entities/relation
        self.entities  = sorted(list(set(self.entities)))
        self.nentity   = len(self.entities)
        self.relations = sorted(list(set(self.relations)))
        self.nrelation = len(self.relations)
        self.ntriples  = len(self.all_triples)
        self.sparsity  = self.ntriples / (self.nentity * self.nentity * self.nrelation)
        self.entity_mapping_dict = {}
        self.relation_mapping_dict = {}

        print('dataset={}, nentity={}, sampled ratio={}, sparsity={}'.format(
            self.dataset, self.nentity, self.sample_ratio, self.sparsity))

        # key:   origin index
        # value: new assigned index
        for idx in range(self.nentity):
            self.entity_mapping_dict[self.entities[idx]] = idx
        for idx in range(self.nrelation):
            self.relation_mapping_dict[self.relations[idx]] = idx

        # get new triples via entitie_mapping_dict
        self.all_new_triples = []
        for (h,r,t) in self.all_triples:
            new_h, new_t = self.entity_mapping_dict[h], self.entity_mapping_dict[t]
            new_r = self.relation_mapping_dict[r]
            new_triples = (new_h, new_r, new_t)
            self.all_new_triples.append(new_triples)

        # shuffle triples
        random.shuffle(self.all_new_triples)

        # split and save data
        self.trainset, self.valset, self.testset = self.splitData()
        self.sampled_dataset = self.saveData(savePath)

    @staticmethod
    def triplesToNxGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.Graph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)
        edges = list(set([(h,t) for (h,r,t) in triples]))
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def triplesToNxDiGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.MultiDiGraph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)

        for (h,r,t) in triples:
            graph.add_edges_from([(h,t)], relation=r)

        return graph
        
    @staticmethod
    def random_walk_induced_graph_sampling(complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])

        iteration   = 1
        growth_size = 2
        check_iters = 100
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node; print(f'==> curr_node: {curr_node}')
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % check_iters == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                    print(f'==> boost seaching, skip to No.{curr_node} node')
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        # sampled_graph = complete_graph.subgraph(Sampled_nodes)
        # return sampled_graph
        return Sampled_nodes

    @staticmethod
    def multi_starts_random_walk_induced_graph_sampling(complete_graph, nodes_to_sample, num_starts):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)

        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes        = len(complete_graph.nodes())
        start_candidate = [random.randint(0, nr_nodes - 1) for i in range(num_starts)]
        Sampled_nodes   = set()

        for idx, index_of_first_random_node in enumerate(start_candidate):
            Sampled_nodes.add(complete_graph.nodes[index_of_first_random_node]['id'])
            iteration           = 1
            growth_size         = 2
            check_iters         = 100
            nodes_before_t_iter = 0
            target_num = int((idx+1) * nodes_to_sample / num_starts)
            curr_node  = index_of_first_random_node

            while len(Sampled_nodes) < target_num:
                edges = [n for n in complete_graph.neighbors(curr_node)]
                index_of_edge = random.randint(0, len(edges) - 1)
                chosen_node = edges[index_of_edge]
                Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
                curr_node = chosen_node
                iteration = iteration + 1

                if iteration % check_iters == 0:
                    if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                        print(f'==> boost seaching, skip to No.{curr_node} node')
                        curr_node = random.randint(0, nr_nodes - 1)
                    nodes_before_t_iter = len(Sampled_nodes)

        return Sampled_nodes

    def splitData(self):
        '''
            split triples with certain ratio stored in self.split_ratio
        '''
        n1 = int(self.ntriples * self.split_ratio[0])
        n2 = int(n1 + self.ntriples * self.split_ratio[1])
        # return self.all_triples[:n1], self.all_triples[n1:n2], self.all_triples[n2:]
        return self.all_new_triples[:n1], self.all_new_triples[n1:n2], self.all_new_triples[n2:]

    def saveData(self, savePath):
        if self.repeat_num == 0:
            if self.num_starts == 1:
                folder = 'sampled_{}_{}'.format(self.dataset, self.sample_ratio)
            else:
                folder = 'sampled_{}_{}_starts_{}'.format(self.dataset, self.sample_ratio, self.num_starts)
        else:
            # folder = 'sampled_{}_{}_rp{}'.format(args['dataset'], self.sample_ratio, self.repeat_num)
            if self.num_starts == 1:
                folder = 'sampled_{}_{}_rp{}'.format(self.dataset, self.sample_ratio, self.repeat_num)
            else:
                folder = 'sampled_{}_{}_starts_{}_rp{}'.format(self.dataset, self.sample_ratio, self.num_starts, self.repeat_num)

        saveFolder = os.path.join(savePath, folder)
        # saveFolder = saveFolder.replace('-', '_')
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        # dataDict = {}
        # dataDict['nentity']       = self.nentity
        # dataDict['nrelation']     = self.nrelation
        # dataDict['train_triples'] = self.trainset
        # dataDict['valid_triples'] = self.valset
        # dataDict['test_triples']  = self.testset
        # dataDict['entity_mapping_dict'] = self.entity_mapping_dict
        # dataDict['relation_mapping_dict'] = self.relation_mapping_dict
        # 
        # dictPath = os.path.join(saveFolder, 'dataset.pkl')
        # print('save to:', dictPath); 
        # # exit()
        # pkl.dump(dataDict, open(dictPath, "wb" ))

        f = open(os.path.join(saveFolder, 'train.del'), 'w')
        for triple in self.trainset: 
            for part in triple:
                f.write(str(part)+"\t")
            f.write("\n")
        f.close()

        f = open(os.path.join(saveFolder, 'valid.del'), 'w')
        for triple in self.valset: 
            for part in triple:
                f.write(str(part)+"\t")
            f.write("\n")
        f.close()

        f = open(os.path.join(saveFolder, 'test.del'), 'w')
        for triple in self.testset: 
            for part in triple:
                f.write(str(part)+"\t")
            f.write("\n")
        f.close()

        ## TODO: There is a bug, but it doesn't matter
        ## 这里直接复制了ids的前若干行，这是不对的，但是对实验没有影响
        fin = open(os.path.join(savePath, self.dataset, 'entity_ids.del'), 'r')
        fout = open(os.path.join(saveFolder, 'entity_ids.del'), 'w')
        for ent in range(self.nentity): 
            line = fin.readline()
            fout.write(line)
        fin.close()
        fout.close()

        fin = open(os.path.join(savePath, self.dataset, 'relation_ids.del'), 'r')
        fout = open(os.path.join(saveFolder, 'relation_ids.del'), 'w')
        for rel in range(self.nrelation): 
            line = fin.readline()
            fout.write(line)
        fin.close()
        fout.close()

        fyaml_read = open(os.path.join(savePath, 'default.yaml'), 'r')
        yaml_data = yaml.full_load(fyaml_read)
        fyaml_read.close()
        yaml_data['dataset']['files.train.size'] = len(self.trainset)
        yaml_data['dataset']['files.valid.size'] = len(self.valset)
        yaml_data['dataset']['files.test.size'] = len(self.testset)
        yaml_data['dataset']['name'] = folder
        yaml_data['dataset']['num_entities'] = self.nentity
        yaml_data['dataset']['num_relations'] = self.nrelation
        fyaml_write = open(os.path.join(saveFolder, 'dataset.yaml'), 'w')
        yaml.dump(yaml_data, fyaml_write)
        fyaml_write.close()

        return folder



def addInverseTriplesForOgblBiokg(triples, nrelation, dictConstant, withNegSamples=False):
    count, true_tail = defaultdict(dictConstant), defaultdict(list)
    inv_heads      = []
    inv_relations  = []
    inv_tails      = []
    inv_head_types = []
    inv_tail_types = []
    head_neg_list  = []
    tail_neg_list  = []

    # counting
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i], triples['relation'][i], triples['tail'][i]
        head_type, tail_type = triples['head_type'][i], triples['tail_type'][i]

        if withNegSamples:
            head_neg, tail_neg = triples['head_neg'][i], triples['tail_neg'][i]
            head_neg_list.append(tail_neg)
            tail_neg_list.append(head_neg)

        # add inverse relation 
        inv_head, inv_relation, inv_tail = tail, relation + nrelation, head
        inv_head_type, inv_tail_type     = tail_type, head_type

        # counting
        count[(head, relation, head_type)] += 1
        count[(inv_head, inv_relation, inv_head_type)] += 1

        # get head/tail peers
        true_tail[(head, relation)].append(tail)
        true_tail[(inv_head, inv_relation)].append(inv_tail)

        inv_heads.append(inv_head)
        inv_relations.append(inv_relation)
        inv_tails.append(inv_tail)
        inv_head_types.append(inv_head_type)
        inv_tail_types.append(inv_tail_type)

    # append inverse triples to train set 
    triples['head']      = np.append(triples['head'],      inv_heads)
    triples['relation']  = np.append(triples['relation'],  inv_relations)
    triples['tail']      = np.append(triples['tail'],      inv_tails)
    triples['head_type'] = np.append(triples['head_type'], inv_head_types)
    triples['tail_type'] = np.append(triples['tail_type'], inv_tail_types)

    if withNegSamples:
        # print(triples['head_neg'].shape, np.array(head_neg_list).shape); exit()
        head_neg_list, tail_neg_list = np.array(head_neg_list), np.array(tail_neg_list)
        print(triples['head_neg'].shape, triples['tail_neg'].shape)
        triples['head_neg'] = np.concatenate([triples['head_neg'], head_neg_list])
        triples['tail_neg'] = np.concatenate([triples['tail_neg'], tail_neg_list])
        print(triples['head_neg'].shape, triples['tail_neg'].shape)

    # return triples, true_tail, count
    return triples


def addInverseTriplesForOgblWikikg2(triples, nrelation, dictConstant, withNegSamples=False):
    count, true_tail = defaultdict(dictConstant), defaultdict(list)
    inv_heads      = []
    inv_relations  = []
    inv_tails      = []
    head_neg_list  = []
    tail_neg_list  = []

    # counting
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i], triples['relation'][i], triples['tail'][i]

        if withNegSamples:
            head_neg, tail_neg = triples['head_neg'][i], triples['tail_neg'][i]
            head_neg_list.append(tail_neg)
            tail_neg_list.append(head_neg)

        # add inverse relation 
        inv_head, inv_relation, inv_tail = tail, relation + nrelation, head

        # counting
        count[(head, relation)] += 1
        count[(inv_head, inv_relation)] += 1

        # get head/tail peers
        true_tail[(head, relation)].append(tail)
        true_tail[(inv_head, inv_relation)].append(inv_tail)

        inv_heads.append(inv_head)
        inv_relations.append(inv_relation)
        inv_tails.append(inv_tail)

    # append inverse triples to train set 
    triples['head']      = np.append(triples['head'],      inv_heads)
    triples['relation']  = np.append(triples['relation'],  inv_relations)
    triples['tail']      = np.append(triples['tail'],      inv_tails)

    if withNegSamples:
        # print(triples['head_neg'].shape, np.array(head_neg_list).shape); exit()
        head_neg_list, tail_neg_list = np.array(head_neg_list), np.array(tail_neg_list)
        print(triples['head_neg'].shape, triples['tail_neg'].shape)
        triples['head_neg'] = np.concatenate([triples['head_neg'], head_neg_list])
        triples['tail_neg'] = np.concatenate([triples['tail_neg'], tail_neg_list])
        print(triples['head_neg'].shape, triples['tail_neg'].shape)

    # return triples, true_tail, count
    return triples



def read_triple(file_path, entity2id, relation2id, nrelation, addInverseRelation=True):
    '''
    Read triples and map them into ids.
    Updates: augment dataset with inverse relation
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            ori_h, ori_r, ori_t = line.strip().split('\t')
            h, r, t = entity2id[ori_h], relation2id[ori_r], entity2id[ori_t]
            triples.append((h, r, t))

            if addInverseRelation:
                triples.append((t, r+nrelation, h))

    return triples

def get_true_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''

    true_tail = defaultdict(list)
    for head, relation, tail in triples:
        true_tail[(head, relation)].append(tail)
        
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))      

    return true_tail

def getIndexingTails(train_true_tail, train_triples):
    tailsByIndex = []
    for head, relation, tail in train_triples:
        tailsByIndex.append(train_true_tail[(head, relation)])

    return tailsByIndex

def getFilteredSamples(triples, all_true_tail, nentity):
    '''
    (1,  tail_index) if invalid (negative triple)
    (-1, tail_index) if valid (exsiting triple)
    '''
    
    # filteredSamples = defaultdict(list)
    filteredSamples = []
    for (head, relation, tail) in tqdm(triples):
        tails              = all_true_tail[(head, relation)]
        filter_bias        = np.ones(nentity)
        filter_bias[tails] *= (-1)
        filter_bias[tail]  = 1
        negative_sample    = [ent for ent in range(nentity)]

        # filteredSamples.append([torch.LongTensor(negative_sample), torch.Tensor(filter_bias)])
        filteredSamples.append(torch.Tensor(filter_bias))

    return filteredSamples

def count_frequency(triples, nrelation, start=4, addInverseRelation=True):
    '''
    Get frequency of a partial triple like (head, relation) or (relation, tail)
    The frequency will be used for subsampling like word2vec
    '''
    count = {}
    for head, relation, tail in triples:
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1

    if not addInverseRelation:
        head_count = {}
        for head, relation, tail in triples:
            if (tail, relation) not in head_count:
                head_count[(tail, relation)] = start
            else:
                head_count[(tail, relation)] += 1

    # countForTriple = {}
    # for head, relation, tail in triples:
    #     tmpCount = count[(head, relation)] + count[(tail, (relation+nrelation)%(2*nrelation))]
    #     countForTriple[(head, relation, tail)] = tmpCount

    countForTriple = []
    for head, relation, tail in triples:
        if addInverseRelation:
            tmpCount = count[(head, relation)] + count[(tail, (relation+nrelation)%(2*nrelation))]
        else:
            tmpCount = count[(head, relation)] + head_count[(tail, relation)]
        countForTriple.append(torch.sqrt(1 / torch.Tensor([tmpCount])))

    return countForTriple

def addInverseRelation(triples, nrelation):
    aug_triples = []
    for (h,r,t) in triples:
        aug_triples.append((t, r+nrelation, h))

    return triples + aug_triples

def addTailNegSamples(triples, all_true_tail, nentity, num=500):
    negSamples = []
    # cnt = 0
    for (head, relation, tail) in tqdm(triples):
        true_tails = list(all_true_tail[(head, relation)])
        # random_idx = list(range(nentity))
        random_idx = list(np.random.randint(0, nentity, num*10))
        random_idx = list(set(random_idx))

        # sample_neg = [tail]
        # while len(sample_neg) < num:
        #     tail = random_idx.pop()
        #     if tail not in true_tails:
        #         sample_neg.append(tail)
            # else:
            #     print(cnt); cnt += 1
            
        sample_neg = [tail] + random_idx[:num]

        negSamples.append(sample_neg)

    negSamples = np.array(negSamples)
    print('==> tail_neg.shape: {}'.format(negSamples.shape))

    return negSamples

def loadData(loadPath):
    if os.path.exists(loadPath):
        file = open(loadPath,'rb')
        datasetInfo = pkl.load(file)
        file.close()
        return datasetInfo["nentity"], datasetInfo["nrelation"], datasetInfo["train_triples"], datasetInfo["valid_triples"], datasetInfo["test_triples"]
    else:
        print("loadData文件不存在", loadPath)
        loali = input()
        return 0,0,[],[],[]

def prepareData(args):
    datasetInfo, entity_dict, all_true_triples = dict(), dict(), dict()
    addInverseRelation_flag = args['addInverseRelation']

    def dictConstant():
        return 4
    
    ###### load OGB datasets ######
    
    if args['dataset'] in ['biokg', 'wikikg2']:
        args['dataset'] = 'ogbl-' + args['dataset']
        from ogb.linkproppred import LinkPropPredDataset
        dataset = LinkPropPredDataset(name = args['dataset'])
        split_edge = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

        if args['dataset'] == 'ogbl-biokg':
            cur_idx = 0
            for key in dataset[0]['num_nodes_dict']:
                entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
                cur_idx += dataset[0]['num_nodes_dict'][key]

            nentity       = sum(dataset[0]['num_nodes_dict'].values())
            nrelation     = int(max(train_triples['relation'])) + 1

            # train_triples = addInverseRelation(train_triples, nrelation)
            # valid_triples = addInverseRelation(valid_triples, nrelation)
            # test_triples  = addInverseRelation(test_triples,  nrelation)
            valid_triples = addInverseTriplesForOgblBiokg(valid_triples, nrelation, dictConstant, withNegSamples=True)
            test_triples  = addInverseTriplesForOgblBiokg(test_triples,  nrelation, dictConstant, withNegSamples=True)
            train_triples = addInverseTriplesForOgblBiokg(train_triples, nrelation, dictConstant)

            # all_true_tail = defaultdict(list)
            train_true_tail = defaultdict(list)
            for i in tqdm(range(len(train_triples['head']))):
                head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
                head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
                train_true_tail[(head, relation)].append(tail)
            for head, relation in train_true_tail:
                train_true_tail[(head, relation)] = np.array(list(set(train_true_tail[(head, relation)])))   

            # indexing_tail = [] # train_true_tail -> indexing_tail
            # for i in tqdm(range(len(train_triples['head']))):
            #     head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
            #     indexing_tail.append(train_true_tail[(head, relation)])

            count = {}; 
            for i in tqdm(range(len(train_triples['head']))):
                head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
                head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]

                if (head, relation) not in count:
                    count[(head, relation)] = 4
                else:
                    count[(head, relation)] += 1

            # indexing to train_count
            # train_count = []
            # for i in tqdm(range(len(train_triples['head']))):
            #     head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
            #     head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
            #     tmpCount = count[(head, relation)] + count[(tail, (relation+nrelation)%(2*nrelation))]
            #     train_count.append(torch.sqrt(1 / torch.Tensor([tmpCount])))

            indexing_tail = train_true_tail
            train_count   = count

        elif args['dataset'] == 'ogbl-wikikg2':
            nentity       = dataset.graph['num_nodes']
            nrelation     = int(max(dataset.graph['edge_reltype'])[0]) + 1
            valid_triples = addInverseTriplesForOgblWikikg2(valid_triples, nrelation, dictConstant, withNegSamples=True)
            test_triples  = addInverseTriplesForOgblWikikg2(test_triples,  nrelation, dictConstant, withNegSamples=True)
            train_triples = addInverseTriplesForOgblWikikg2(train_triples, nrelation, dictConstant)

            # train_true_tail = defaultdict(list)
            # for i in tqdm(range(len(train_triples['head']))):
            #     head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
            #     train_true_tail[(head, relation)].append(tail)
            # for head, relation in train_true_tail:
            #     train_true_tail[(head, relation)] = np.array(list(set(train_true_tail[(head, relation)]))) 
            
            count = {}; 
            for i in tqdm(range(len(train_triples['head']))):
                head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
                if (head, relation) not in count:
                    count[(head, relation)] = 4
                else:
                    count[(head, relation)] += 1
            
            # indexing_tail = train_true_tail
            indexing_tail = {}
            train_count   = count

    
            # too costing
            # indexing_tail = []
            # for i in tqdm(range(len(train_triples['head']))):
            #     head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
            #     indexing_tail.append(train_true_tail[(head, relation)])

            # too costing
            # indexing to train_count
            # train_count = []
            # for i in tqdm(range(len(train_triples['head']))):
            #     head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
            #     tmpCount = count[(head, relation)] + count[(tail, (relation+nrelation)%(2*nrelation))]
            #     train_count.append(torch.sqrt(1 / torch.Tensor([tmpCount])))

    
    ###### load other datasets: wn18(rr) fb15k(237) ######
    elif args['dataset'] in ['wn18', 'wnrr', 'FB15k', 'fb15k-237', 'YAGO3_10', 'umls', 'kinship', 'family', 'toy']:
        with open(os.path.join("./data", args['dataset'], 'entity_ids.del')) as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join("./data", args['dataset'], 'relation_ids.del')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        nentity           = len(entity2id)
        nrelation         = len(relation2id)
        
        # augment train data via inverse relation
        train_triples     = read_triple(os.path.join("./data", args['dataset'], 'train.txt'), entity2id, relation2id, nrelation, addInverseRelation_flag)
        valid_triples     = read_triple(os.path.join("./data", args['dataset'], 'valid.txt'), entity2id, relation2id, nrelation, addInverseRelation_flag)
        test_triples      = read_triple(os.path.join("./data", args['dataset'], 'test.txt'),  entity2id, relation2id, nrelation, addInverseRelation_flag)

        # All true triples
        all_true_triples  = set(train_triples + valid_triples + test_triples)

        # get head/tail peers
        train_true_tail   = get_true_tail(train_triples)
        all_true_tail     = get_true_tail(all_true_triples)
        indexing_tail     = getIndexingTails(train_true_tail, train_triples)
        
        # counting
        train_count       = count_frequency(train_triples, nrelation, addInverseRelation=addInverseRelation_flag)

        # get negative samples for evaluation
        valid_negSamples  = getFilteredSamples(valid_triples, all_true_tail, nentity)
        test_negSamples   = getFilteredSamples(test_triples,  all_true_tail, nentity)
        # train_negSamples  = getFilteredSamples(train_triples, all_true_tail, nentity) # too costing

    elif 'sampled_ogbl_wikikg2' in args['dataset']:
        
        # loading synthetic data / other datasets
        savePath = os.path.join("./data", args['dataset'], 'dataset.pkl')
        nentity, nrelation, train_triples, valid_triples, test_triples = loadData(savePath)
        nentity, nrelation = int(nentity), int(nrelation)

        # add inverse relation manully
        if addInverseRelation_flag:
            train_triples = addInverseRelation(train_triples, nrelation)
            valid_triples = addInverseRelation(valid_triples, nrelation)
            test_triples  = addInverseRelation(test_triples,  nrelation)

        # All true triples
        all_true_triples  = set(train_triples + valid_triples + test_triples)

        # get head/tail peers
        train_true_tail   = get_true_tail(train_triples)
        all_true_tail     = get_true_tail(all_true_triples)
        indexing_tail     = getIndexingTails(train_true_tail, train_triples)

        # add tail_neg for each val/test triple
        valid_negSamples = addTailNegSamples(valid_triples, all_true_tail, nentity, num=500)
        test_negSamples  = addTailNegSamples(test_triples,  all_true_tail, nentity, num=500)
        
        # counting
        train_count       = count_frequency(train_triples, nrelation, addInverseRelation=addInverseRelation_flag)

    else: 
        
        # loading synthetic data / other datasets
        savePath = os.path.join("./data", args['dataset'], 'dataset.pkl')
        nentity, nrelation, train_triples, valid_triples, test_triples = loadData(savePath)
        nentity, nrelation = int(nentity), int(nrelation)

        # add inverse relation manully
        if addInverseRelation_flag:
            train_triples = addInverseRelation(train_triples, nrelation)
            valid_triples = addInverseRelation(valid_triples, nrelation)
            test_triples  = addInverseRelation(test_triples,  nrelation)

        # All true triples
        all_true_triples  = set(train_triples + valid_triples + test_triples)

        # get head/tail peers
        train_true_tail   = get_true_tail(train_triples)
        all_true_tail     = get_true_tail(all_true_triples)
        indexing_tail     = getIndexingTails(train_true_tail, train_triples)
        
        # counting
        train_count       = count_frequency(train_triples, nrelation, addInverseRelation=addInverseRelation_flag)

        # get negative samples for val/test set
        if 'sampled_ogbl_wikikg2' in args['dataset']:
            valid_negSamples  = [] # too costing
            test_negSamples   = []
        else:
            # train_negSamples  = getFilteredSamples(train_triples, all_true_tail, nentity)
            valid_negSamples  = getFilteredSamples(valid_triples, all_true_tail, nentity)
            test_negSamples   = getFilteredSamples(test_triples,  all_true_tail, nentity)


    datasetInfo['datasetName']      = args['dataset'] 
    datasetInfo['entity_dict']      = entity_dict
    datasetInfo['nentity']          = nentity
    datasetInfo['nrelation']        = nrelation
    datasetInfo['train_triples']    = train_triples
    datasetInfo['valid_triples']    = valid_triples
    datasetInfo['test_triples']     = test_triples
    datasetInfo['indexing_tail']    = indexing_tail
    # datasetInfo['train_true_tail']  = train_true_tail
    datasetInfo['train_count']      = train_count
    datasetInfo['train_len']        = len(train_triples['head']) if args['dataset'] in ['ogbl-biokg', 'ogbl-wikikg2'] \
                                        else len(train_triples)

    if args['dataset'] not in ['ogbl-biokg', 'ogbl-wikikg2']:
        datasetInfo['valid_negSamples'] = valid_negSamples
        datasetInfo['test_negSamples']  = test_negSamples
        # datasetInfo['train_negSamples'] = train_negSamples
        datasetInfo['all_true_tail']    = all_true_tail

    return datasetInfo
