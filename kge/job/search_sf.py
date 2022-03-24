import os
from kge.job import AutoSearchJob, Job
from kge import Config
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from torch.optim import SGD

class SFSearchJob(AutoSearchJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.num_trials = self.config.get("sf_search.num_trials")
        # self.structure_genarator_mode = self.config.get("sf_search.structure_genarator_mode")
        # self.prob = self.config.get("sf_search.prob")
        self.K = self.config.get("sf_search.K")
        self.structure_genarator = SturctGenarator(self.K)

        if self.__class__ == SFSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    def register_trial(self, parameters=None):
        parameters, trial_id = self.structure_genarator.get_next_trail()
        # parameters: 优化参数的字典
        # trail_id: 训练次数, int
        return parameters, trial_id


    def register_trial_result(self, trial_id, parameters, trace_entry):
        # trace_entry["metric_value"]: MRR, float
        # parameters: 优化参数的字典
        self.structure_genarator.update_genarator(parameters, trace_entry["metric_value"])


    def get_best_parameters(self):
        pass


class SturctGenarator():
    def __init__(self, K=4, topk=10, MAX_POPU=5, N_CAND = 100):
        self.K = K
        self.topk = topk
        self.MAX_POPU = MAX_POPU
        self.N_CAND = N_CAND

        self.struct_obj = StructSpace(self.K)  
        self.predictor = Predictor(topk=self.topk, K=self.K)
        self.model_perf = [0]
        self.model_struc = [[0]]
        self.model_cand = []
        self.best_model = []
        self.best_mrr = 0
        self.model_cand += self.struct_obj.populations
        self.turn = list(range(len(self.model_cand)))
        
        self.next_turn = len(self.turn)
        self.init_num = len(self.turn)-1
        self.struct_by_predictor = 0


    def get_next_trail(self):
        # try:
        print("before", self.model_cand, self.turn)
        next_A = self.model_cand.pop(0)
        parameters = {"autosf.A": next_A}
        trial_id = self.turn.pop(0)
        print("get_next_trial", parameters, trial_id)
        # except IndexError: 
        #     print("No more struct candidate")
        #     print("best model is: ", self.best_model)
        #     print("best mrr is: ", self.best_mrr)
        
        return parameters, trial_id


    def update_genarator(self, A_dict, mrr):
        struct_A = A_dict["autosf.A"]
        self.struct_obj.get_pred_data(struct_A, mrr)
        if mrr > self.best_mrr:
            self.best_mrr = mrr
            self.best_model = struct_A
        if len(self.model_perf) < self.MAX_POPU:
            self.model_perf.append(mrr)
            self.model_struc.append(tuple(struct_A))
        elif mrr > min(self.model_perf):
            idx = np.argmin(self.model_perf)
            # print('\t\tupdate popu:', model_struc[idx], model_perf[idx], '--->', struct, mrr)
            self.model_perf[idx] = mrr
            self.model_struc[idx] = struct_A    

        if(self.init_num > 0):
            self.init_num -= 1
        elif(self.struct_by_predictor == 0):
            struct_cand = self.struct_obj.gen_new_struct(self.model_struc, self.N_CAND)
            features = []
            for struct in struct_cand:
                features.append(self.struct_obj.gen_features(struct))
            top_idx = self.predictor.evaluate(features, self.struct_obj.pred_x, self.struct_obj.pred_y)
            self.model_cand = self.model_cand + np.array(struct_cand)[top_idx].tolist()
            self.turn = self.turn + list(range(self.next_turn, self.next_turn + len(top_idx)))
            self.next_turn += len(top_idx)
            self.struct_by_predictor = len(top_idx)-1
        else:
            self.struct_by_predictor -= 1

        



class StructSpace:
    def __init__(self, K):
        self.pred_x = []
        self.pred_y = []
        self.K = K
        self.noise = 0.02
        self.back_ups = set()
        self.signs = self.get_sign()
        self.populations = self.gen_base_struct()

    def get_sign(self,):
        base = [2**i for i in range(self.K)]
        signs = []
        for i in range(2**self.K):
            sign = [1] * self.K
            for j in range(self.K):
                if i & base[j] > 0:
                    sign[j] = -1
            signs.append(sign) 
        return signs 

    def gen_base_struct(self,):
        # generate structures with K nonzero blocks
        indices = list(range(self.K))
        all_struct = [] 
        for perm in list(itertools.permutations(indices)):
            struct = [0] * self.K**2
            for i in range(self.K):
                struct[i*self.K + perm[i]] = i+1
                if not self.filt(struct, self.back_ups):
                    continue
                all_struct.append(struct)
                self.back_ups = self.back_ups.union(self.get_equivalence(struct))
        return all_struct

    # 过滤掉degenerate和equivalence
    def filt(self, struct, H_set):
        matrix = np.reshape(struct, (self.K, self.K))
        
        # not full rank
        if np.linalg.det(matrix) == 0:
            return False

        rela = np.ones((self.K,))

        # all the K components in r should be covered
        for idx in np.nonzero(struct)[0]:
            r = abs(struct[idx]) - 1 
            rela[r] = 0 
        
        if np.sum(rela) > 0:
            return False

        # equivalence
        if tuple(struct) in H_set:
            return False
        return True

    # 生成所有的equivalence
    def get_equivalence(self, struct):
        indices = list(range(self.K))
        np_struct = np.array(struct)
        all_struct = set()
        for perm_rel in list(itertools.permutations(indices)):
            for perm_ent in list(itertools.permutations(indices)):
                for sign in self.signs:
                    new_struct = [0] * (self.K**2)
                    for idx in np.nonzero(np_struct)[0]:
                        h = perm_ent[idx // self.K]
                        t = perm_ent[idx % self.K]
                        r = perm_rel[abs(np_struct[idx]) - 1]
                        s = sign[abs(np_struct[idx]) - 1]
                        new_struct[h*self.K+t] = s * np.sign(np_struct[idx]) * (r+1)
                    all_struct.add(tuple(new_struct))

        return all_struct


    def get_pred_data(self, struct, perf, one_hot=False):
        # SRF

        if one_hot:
            features = [0] * (self.K**2 * (self.K+2))
            for idx in np.nonzero(struct)[0]:
                r = abs(struct[idx])
                s = np.sign(struct[idx])
                features[idx*(self.K+2)+r-1] = 1
                if s>0:
                    features[idx * (self.K+2) + self.K] = 1
                else:
                    features[idx * (self.K+2) + self.K+1] = 1
            self.pred_x.append(features)
            self.pred_y.append(perf + self.noise*(2*np.random.random()-1))
            return
        else:
            K = self.K
            values = [list(range(-K, K+1)) for i in range(K)]
    
            SRF = np.zeros((K*(K+1),))
            for assign in itertools.product(*values):
                a = K-np.count_nonzero(assign)
                if a == K:
                    continue
                b = len(list(set(np.abs(assign)) - {0}))
                k = K*a - ((a-1)*a) // 2 + b - 1
                k0 = (K * (K+1)) // 2

                matrix = np.zeros((K, K))
                for idx in np.nonzero(struct)[0]:
                    h = idx // K
                    t = idx % K
                    r = np.sign(struct[idx]) * assign[abs(struct[idx]) - 1]
                    matrix[h][t] = r
                if np.sum(abs(matrix)) == 0:
                    continue
                if np.sum(abs(matrix - matrix.T)) < 0.0001:
                    SRF[k] = 1
                if np.sum(abs(matrix + matrix.T)) < 0.0001 and np.sum(abs(matrix - np.diag(matrix))) > 0:
                    SRF[k+k0] = 1

            self.pred_x.append(SRF)
            self.pred_y.append(perf + self.noise*(2*np.random.random()-1))      # 这个噪声加了干嘛 ???
            return

    def gen_features(self, struct, one_hot=False):

        if one_hot:
            features = [0] * (self.K**2 * (self.K+2))
            for idx in np.nonzero(struct)[0]:
                r = abs(struct[idx])
                s = np.sign(struct[idx])
                features[idx*(self.K+2)+r-1] = 1
                if s>0:
                    features[idx * (self.K+2) + self.K] = 1
                else:
                    features[idx * (self.K+2) + self.K+1] = 1
            return features

        K = self.K
        values = [list(range(-K, K+1)) for i in range(K)]
    
        SRF = np.zeros((K*(K+1),))
        for assign in itertools.product(*values):
            a = K - np.count_nonzero(assign)
            b = len(list(set(np.abs(assign)) - {0}))
            if a== K:
                continue
            k = K*a - (a-1)*a // 2 + b - 1
            k0 = K * (K+1) // 2

            matrix = np.zeros((K, K))
            for idx in np.nonzero(struct)[0]:
                h = idx // K
                t = idx % K
                r = np.sign(struct[idx]) * assign[abs(struct[idx]) - 1]
                matrix[h][t] = r
            if np.sum(abs(matrix)) == 0:
                continue
            if np.sum(abs(matrix - matrix.T)) < 0.0001:
                SRF[k] = 1
            if np.sum(abs(matrix + matrix.T)) < 0.0001 and np.sum(abs(matrix - np.diag(matrix))) > 0:
                SRF[k+k0] = 1
        return SRF


    def gen_new_struct(self, parents, N_CAND):
        results = []
        current_set = set()
        failed = 0
        while(len(results)<N_CAND and failed<100):
            # crossover
            if np.random.random() < 0.5: 
                p1, p2 = np.random.choice(len(parents), size=(2,), replace = False)
                P1 = parents[p1]
                P2 = parents[p2]
                new_struct = []
                for i in range(self.K**2):
                    if np.random.random() < 0.5:
                        new_struct.append(P1[i])
                    else:
                        new_struct.append(P2[i])

                if np.random.random() < 0.2:
                    new_struct = self.mutate(new_struct)
            # mutation
            else:             
                p = np.random.choice(len(parents))
                struct = parents[p]
                new_struct = self.mutate(struct)

            if self.filt(new_struct, self.back_ups.union(current_set)):
                results.append(new_struct)
                new_equiv = self.get_equivalence(new_struct)
                current_set = current_set.union(new_equiv)
                failed = 0
            else: 
                failed += 1
        return results
            
    def mutate(self, old_struct):
        new_struct = list(old_struct)
        for i in range(len(old_struct)):
            if np.random.random() < 2/(self.K**2):
                new_struct[i] = np.random.choice(2*self.K+1) - self.K
        return new_struct

class Regressor(nn.Module):
    def __init__(self,K):
        super(Regressor, self).__init__()
        #length = 118
        length = K*(K+1)
        hid = 2
        self.layer1 = nn.Linear(length, hid)
        self.layer2 = nn.Linear(hid, 1)
        self.act = nn.ReLU()

    def forward(self, x, dropout=0):
        drop = nn.Dropout(dropout)
        x = drop(x)
        outs = self.layer1(x)
        outs = self.act(outs)
        outs = self.layer2(outs).squeeze()
        return outs

class Predictor(object):
    def __init__(self, topk=5, K=4):
        self.model = Regressor(K).cuda()
        self.loss = nn.L1Loss()
        self.batch_size = 12
        self.topk = topk
        self.optim = SGD(self.model.parameters(), lr=0.01, weight_decay=0.1)

    def evaluate(self, candidates, X, Y, dropout=0.0):
        self.model.train()
        n = len(Y)
        train_x = torch.FloatTensor(np.array(X)).cuda()
        train_y = torch.FloatTensor(np.array(Y)).cuda()
        n_iter = 200
        for i in range(n_iter):
            self.model.zero_grad()
            idx = np.random.choice(n, self.batch_size)
            x_batch = train_x[idx]
            y_batch = train_y[idx]
            y_p = self.model(x_batch, dropout)
            loss = self.loss(y_p, y_batch)
            loss.backward()
            self.optim.step()

        test_X = torch.FloatTensor(np.array(candidates)).cuda()
        self.model.eval()
        scores = self.model(test_X, 0).cpu().data.numpy()
        top_index = scores.argsort()[-self.topk:][::-1]
        #print(scores)

        top_y = scores[top_index]
        #print(top_index, top_y)
        print(top_y)
        return top_index


    def get_scores(self, vectors):
        scores = self.model(vectors)
        return scores.data.numpy()

    def train(self, x, y, batch_size=128, dropout=0.0, n_iter=300):
        n = y.size(0)
        batch_size = max(n // 8, 1)
        for i in range(n_iter):
            self.model.zero_grad()
            idx = np.random.choice(n, batch_size)
            x_batch = x[idx]
            y_batch = y[idx]
            y_p = self.model(x_batch, dropout)
            loss = self.loss(y_p, y_batch)
            loss.backward()
            self.optim.step()

