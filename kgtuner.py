import os
from unittest import result
import yaml
import time
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Search Configs and CUDA Selection for kgtuner',
        usage='kgtuner.py [<args>] [-h | --help]'
    )
    parser.add_argument('--config', type=str, default='kgtuner_fb15k237_40in200', help='config file in /kgtuner')
    parser.add_argument('--device', type=str, default=' ', help='select a device, e.g. cuda:1')
    return parser.parse_args(args)

args = parse_args()
config_file = args.config

fin = open(os.path.join('kgtuner', config_file+'.yaml'))
kgtuner_main = yaml.full_load(fin)
kgtuner1_config = kgtuner_main['kgtuner1']
kgtuner2_config = kgtuner_main['kgtuner2']
fin.close()

result_folder = 'kgtuner/results/' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + config_file
kgtuner1_folder = result_folder + '/kgtuner1'
kgtuner2_folder = result_folder + '/kgtuner2'
kgtuner1_yaml_path = kgtuner1_folder + '/kgtuner1.yaml'
kgtuner2_yaml_path = kgtuner2_folder + '/kgtuner2.yaml'
kgtuner1_config['ax_search']['folder_path'] = result_folder        # TODO: ax改成kgtuner1
kgtuner2_config['kgtuner2']['folder_path'] = result_folder

os.system("mkdir " + result_folder)
os.system("mkdir " + kgtuner1_folder)
os.system("mkdir " + kgtuner2_folder)

f_kgtuner1 = open(kgtuner1_yaml_path, 'w')
yaml.dump(kgtuner1_config, f_kgtuner1)
f_kgtuner1.close()

f_kgtuner2 = open(kgtuner2_yaml_path, 'w')
yaml.dump(kgtuner2_config, f_kgtuner2)
f_kgtuner2.close()

if args.device==' ':
    os.system('kge start ' + kgtuner1_yaml_path)
    os.system('kge start ' + kgtuner2_yaml_path)
else:
    os.system('kge start ' + kgtuner1_yaml_path + ' --job.device ' + args.device)
    os.system('kge start ' + kgtuner2_yaml_path + ' --job.device ' + args.device)

