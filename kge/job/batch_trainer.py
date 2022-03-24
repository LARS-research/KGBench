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

class BatchTrainer(AutoSearchJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.num_trials = self.config.get("batch_trainer.num_trials")
        # self.structure_genarator_mode = self.config.get("sf_search.structure_genarator_mode")
        # self.prob = self.config.get("sf_search.prob")
        # self.K = self.config.get("sf_search.K")
        # self.structure_genarator = SturctGenarator(self.K)

        self.dataset_info = self.config.get("dataset")
        self.data_choice = self.config.get("batch_trainer.data_choice")
        self.record_of_mrr = []
        
        self.idx_of_trial1 = [1,3,4,6,7,9,10,11,14,16,64,55,19,95,86,45,72,69,39,92,60,22,23,34,46
            ] if self.data_choice=='wnrr' else [55,64,1,11,14,15,17,19,20,25,28,34,42,45,50,54,59,70,75,67,71,74,6,10,43]
        self.idx_of_trial2 = [1,64,86,19,55,45,72,95,22,23,31,34,4,9,16,32,77,99,7,14,26,33,50,39,44
            ] if self.data_choice=='wnrr' else [55,1,11,14,17,19,20,25,28,34,42,45,50,54,59,6,48,24,43,4,15,9,32,35,51]
        self.mrr_of_whole = [0.46355,0.40341,0.421,0.39277,0.34689,0.42976,0.43678,0.4278,0.3978,0.42627,0.463,0.4604,0.45118,0.45161,0.45848,0.44946,0.44515,0.17836,0.27581,0.29198,0.2677,0.43659,0.43176,0.43977,0.43359,
            0.46017,0.46065,0.4618,0.45807,0.45913,0.44309,0.44911,0.44443,0.43694,0.43031,0.43565,0.4332,0.42424,0.42918,0.42691,0.42007,0.42289,0.4273,0.35819,0.39356,0.30237,0.37057,0.33945,0.27445,0.29017
            ] if self.data_choice=='wnrr' else [0.33692,0.35033,0.29293,0.21796,0.22931,0.23349,0.20054,0.25614,0.21339,0.20788,0.22593,0.23662,0.21377,0.27264,0.22273,0.20247,0.20563,0.2304,0.21614,0.19088,0.1895,0.18794,0.19545,0.15882,0.18653,
            0.33971,0.29107,0.22372,0.21891,0.20051,0.26285,0.2091,0.20517,0.21654,0.23044,0.22724,0.2612,0.22717,0.20173,0.2072,0.19668,0.19953,0.18639,0.18664,0.17183,0.17293,0.16789,0.13971,0.15282,0.16436]
        
        # self.idx_of_trial1 = list(range(25))
        # self.idx_of_trial2 = list(range(25))
        # self.mrr_of_whole = list(range(4))
        self.trial_id = 0
        config_path1 = "20220218-222536-ax_search_wnrr_sf0" if self.data_choice=='wnrr' else "20220218-222735-ax_search_fb15k237_sf0"
        config_path2 = "20220218-222445-ax_search_wnrr_sf1" if self.data_choice=='wnrr' else "20220218-222600-ax_search_fb15k237_sf1"
        self.configload_path1 = os.path.join("local", "experiments",config_path1)
        self.configload_path2 = os.path.join("local", "experiments",config_path2)
        if self.__class__ == BatchTrainer:
            for f in Job.job_created_hooks:
                f(self)

    def register_trial(self, parameters=None):
        if self.trial_id < 25:
            raw_config_folder = ('0000' if self.idx_of_trial1[self.trial_id]<10 else'000')+str(self.idx_of_trial1[self.trial_id])
            fin = open(os.path.join(self.configload_path1,raw_config_folder,'config.yaml'))
        else:
            raw_config_folder = ('0000' if self.idx_of_trial2[self.trial_id-25]<10 else'000')+str(self.idx_of_trial2[self.trial_id-25])
            fin = open(os.path.join(self.configload_path2,raw_config_folder,'config.yaml'))
        parameters = yaml.full_load(fin)
        fin.close()
        parameters['dataset'] = self.dataset_info
        self.trial_id += 1
        trial_id = self.trial_id
        return parameters, trial_id


    def register_trial_result(self, trial_id, parameters, trace_entry):
        # trace_entry["metric_value"]: MRR, float
        # parameters: 优化参数的字典
        self.record_of_mrr.append(trace_entry["metric_value"])

    def get_best_parameters(self):
        df = pd.DataFrame({'whole': self.mrr_of_whole, 'sampled': self.record_of_mrr})
        srcc = df.corr('spearman')['whole']['sampled']
        print("srcc: ", srcc)
        self.config.log(
                        "mrr on sampled graph: {}\n mrr on whole graph:{}\n srcc:{}".format(
                            self.record_of_mrr, self.mrr_of_whole, srcc
                        )
                    )

