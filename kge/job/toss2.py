import os
from kge.job import AutoSearchJob, Job
from kge import Config
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from torch.optim import SGD
import yaml
import pandas as pd
import numpy as np

class TOSS2(AutoSearchJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.num_trials = self.config.get("toss2.topK")
        self.folder_path = self.config.get("toss2.folder_path")
        self.trial = 0
        self.mrr_record = []
        # self.structure_genarator_mode = self.config.get("sf_search.structure_genarator_mode")
        # self.prob = self.config.get("sf_search.prob")
        # self.K = self.config.get("sf_search.K")
        # self.structure_genarator = SturctGenarator(self.K)
        if self.__class__ == TOSS2:
            for f in Job.job_created_hooks:
                f(self)

    def register_trial(self, parameters=None):
        fin = open(self.folder_path + "/toss1/exp" + str(self.trial) + '.yaml')
        config = yaml.full_load(fin)
        fin.close()
        if 'train.batch_size' in config.keys():
            config['train.batch_size'] = min(1024, config['train.batch_size'] * 2)
        if 'lookup_embedder.dim' in config.keys():
            config['lookup_embedder.dim'] = min(1024, config['lookup_embedder.dim'] * 2)
        self.trial += 1
        return config, self.trial-1


    def register_trial_result(self, trial_id, parameters, trace_entry):
        # trace_entry["metric_value"]: MRR, float
        # parameters: 优化参数的字典
        self.mrr_record.append(trace_entry["metric_value"])

    def get_best_parameters(self):
        fout = open(self.folder_path + '/toss2/result.yaml', 'w')
        best_trial = int(np.argmax(self.mrr_record))
        result = {'mrr': self.mrr_record , 'beat_trial': best_trial}
        yaml.dump(result, fout)
        fout.close()
