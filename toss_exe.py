import os
from unittest import result
import yaml
import time

config_file = 'toss_fb15k237_40in200'

fin = open(os.path.join('toss', config_file+'.yaml'))
toss_main = yaml.full_load(fin)
toss1_config = toss_main['toss1']
toss2_config = toss_main['toss2']
fin.close()

result_folder = 'toss/results/' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + config_file
toss1_folder = result_folder + '/toss1'
toss2_folder = result_folder + '/toss2'
toss1_yaml_path = toss1_folder + '/toss1.yaml'
toss2_yaml_path = toss2_folder + '/toss2.yaml'
toss1_config['ax_search']['folder_path'] = result_folder        # TODO: ax改成toss1
toss2_config['toss2']['folder_path'] = result_folder

os.system("mkdir " + result_folder)
os.system("mkdir " + toss1_folder)
os.system("mkdir " + toss2_folder)

f_toss1 = open(toss1_yaml_path, 'w')
yaml.dump(toss1_config, f_toss1)
f_toss1.close()

f_toss2 = open(toss2_yaml_path, 'w')
yaml.dump(toss2_config, f_toss2)
f_toss2.close()

os.system('kge start ' + toss1_yaml_path)
os.system('kge start ' + toss2_yaml_path)


