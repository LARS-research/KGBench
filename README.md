# READ ME
新增了一种模型(model)AutoSF和相关的结构搜索方法(job)，新增了超参搜索方法TOSS。


## Quick Start


```sh
# 配置环境
cd KGBench
pip install -e .
cd..

# 下载以及预处理部分数据集
cd data
sh download_all.sh
cd ..

# 在dataset toy上训练模型，device选择cuda:0，超参config位于examples/toy-complex-train.yaml
kge start examples/toy-complex-train.yaml --job.device cuda:0

```

训练命令主要使用kge start，除此以外还有其他一些命令，可以在 ReadMe0.md 中进行查询。

### Hyperparameter Configuration

一般将config写成 .yaml文件，写在 ./examples 下，利用命令 kge start 开始训练，默认的超参数和细节解释在[config-default.yaml](kge/config-default.yaml)（训练过程的超参数，例如lr，batch_size等）和[lookup_embedder.yaml](kge/model/embedder/lookup_embedder.yaml)（查找表的超参数，例如dropout，dim等），模型相关的超参数和细节解释在 ./kge/model、下对应模型的yaml文件中，例如[autosf.yaml](kge/model/autosf.yaml)

### Dataset

Dataset保存在文件夹./data/下，如果要添加更多的数据集主要需要六个文件：

dataset.yaml：数据集的相关信息，具体内容参考已有的数据集，最重要的信息例如有以下几个：

```yaml
dataset:
  files.test.filename: test.del			# 测试集的文件名
  files.test.size: 2680					# 测试集大小
  files.test.type: triples				# 测试集组成方式，如果类似biokg有负样本需要另外设置，细节参考biokg
  files.train.filename: train.del
  files.train.size: 48215
  files.train.type: triples
  files.valid.filename: valid.del
  files.valid.size: 2678
  files.valid.type: triples
  name: sampled_fb15k-237_0.2			# 数据集名称
  num_entities: 2901					# entity的数目（可以设成-1）
  num_relations: 225					# relation的数目（可以设成-1）

```

train.del：训练集的index表示（一般为 $ \#train\times3 $）

valid.del：验证集的index表示（一般为 $ \#valid\times3 $）

test.del：测试集的index表示（一般为 $ \#test\times3 $）

entity_ids.del：entity的index与对应的实体（一般为 $ \#entity\times2 $）

relation_ids.del：relation的index与对应的关系（一般为 $ \#relation\times2 $）



## AutoSF

### 训练
在config中设置model为autosf，另外需要设置autosf的相关参数A（布局）和K（分块数目），如下：

```yaml
model: autosf
autosf:
  A: [1,0,3,0, 0,2,0,4, -3,0,1,0, 0,-4,0,2]
  K: 4
```

autosf的相关参数以及默认设置参见 /kge/model/autosf.yaml

### 搜索

需要设置 job.type 为search，search.type为sf，model为autosf，另外需要设置sf_search的相关参数num_trials（搜索轮数）和K（分块数目），如下：

```yaml
job.type: search
search.type: sf

model: autosf

sf_search:
  num_trials: 300
  K: 4

```

sf_search的相关参数以及默认设置位于 /kge/config-default.yaml 之中



## KGTuner（TOSS）

### 子图采样

在参数的.yaml文件中设置job.type为搜索，search.type为subgraph_sample，在subgraph_sample.sampled_ratio中设置采样率，例如

```yaml
job.type: search
dataset.name: biokg
search.type: subgraph_sample

subgraph_sample:
  sampled_ratio: 0.2

```



### 搜索

将设置文件写到 /toss 文件夹中，运行 toss_exe.py 文件（修改config_file为设置文件），结果保存在 /toss/results当中。

目前toss第一阶段采用的是ax的搜索方法，所以toss1中要设置 search.type 为ax，以及设置ax相关的参数，例如：

```yaml
toss1:
  job.type: search
  dataset.name: sampled_wnrr_0.2
  model: complex
  
  search.type: ax
  ax_search:
    topK: 30
    record_topK: true
    num_trials: 200
    num_sobol_trials: 70
    parameters:
      - name: train.batch_size
        type: choice
        values: [128, 256, 512]
```

注意ax_search中要设置 record_topK 为 true，用于保存搜索结果，topK为toss1保存下来的参数的数目；dataset.name 要设置成子图的数据集，job.type设成search，search.type设成ax。

注意toss1和toss2中除了搜索之外的固定的超参数需要在toss1和toss2中都写一次。

toss2中需要设置dataset.name为原有的数据集，search.type为 toss2，toss2的相关参数 topK 和 num_trials 为第二阶段在原有数据集上训练的超参数的组数，请将这两者和 toss1 中的 topK 都设为一样。例如：

```yaml
toss2:
  job.type: search
  dataset.name: wnrr
  model: complex
  
  search.type: toss2

  toss2:
    topK: 30
    num_trials: 30
    
```






## 关于ogb两个数据集

search任务注意把 valid.metric 设成mrr而不是mrr_filtered

如果valid或test中指定负采样请修改dataset中如下：

```yaml
dataset:
  files.test.type: triples_and_negs
  files.test.neg_for_head: 500			# head负样本数目
  files.test.neg_for_tail: 500			# tail负样本数目
  files.valid.type: triples_and_negs
  files.test.neg_for_head: 500			# head负样本数目
  files.test.neg_for_tail: 500			# tail负样本数目
```

目前这两个数据集的evaluate只支持autosf和ComplEX模型。



以上部分如有问题和bug请联系李霖lli18@mails.tsinghua.edu.cn