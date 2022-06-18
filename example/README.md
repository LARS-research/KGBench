# README
Here is the instruction of AutoBLM, KGTuner, Relation Prediction and Node Piece.
## AutoBLM: Scoring Function Search

Original [paper](https://ieeexplore.ieee.org/document/9729658) and [code](https://github.com/AutoML-Research/AutoSF).

You can conduct the scoring function search algorithm AutoBLM by setting `search.type` as `blm` and setting model as `autoblm`. For example, you have a [blm_search_easy.yaml](example/FB15k-237/blm_search_easy.yaml) config file:

```yaml
job.type: search
search.type: blm
dataset.name: fb15k-237

model: autoblm

blm_search:
  num_trials: 300
  K: 4
```
Then you can start blm search on fb15k-237 easily with other hyperparameters set as default: 

```bash
kgbench start example/FB15k-237/blm_search_easy.yaml
```
We search blm on ogbl-biokg and ogbl-wikikg2 using [biokg_blm_search.yaml](example/biokg/biokg_blm_search.yaml) and [wikikg2_blm_search.yaml](example/wikikg2/wikikg2_blm_search.yaml).

See `blm_search` in [config-default.yaml](kgbench/config-default.yaml) for more hyperparameters used in search.

For a given model structure searched by AutoBLM,  you can set `model` as `autoblm` and `autoblm.A` as the given structure for your further study. See [autoblm.yaml](kgbench/model/autoblm.yaml) for more hyperparameters used in model autoblm.



## KGTuner: Hyperparameter Search

Original [paper](https://aclanthology.org/2022.acl-long.194.pdf) and [code](https://github.com/AutoML-Research/KGTuner).

KGTuner has two stages, searching on the subgraph for the first stage and on the original graph for the second stage. In the first stage, use subgraph sampling job to do downsampling. For example, 

```yaml
job.type: search
dataset.name: wnrr
search.type: subgraph_sample
model: complex
subgraph_sample.sampled_ratio: 0.2
```

The configuration file should be saved in the folder *kgtuner*  and use kgtuner.py under the main directory to start KGTuner search. For example, you have such a config file in kgtuner which titled as [example_for_kgtuner.yaml](kgtuner/example_for_kgtuner.yaml): 

```yaml
stage1:
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
        
stage2:
  job.type: search
  dataset.name: wnrr
  model: complex
  
  search.type: kgtuner2

  kgtuner2:
    topK: 30
    num_trials: 30
```

Then you can run python commond in the terminal to start search, for example,

```bash
python kgtuner.py --config example_for_kgtuner.yaml --device cuda:1
```



## Node Piece and Relation Prediction

We have also implemented [Node Piece](https://arxiv.org/abs/2106.12144) and [Relation Prediction](https://openreview.net/pdf?id=Qa3uS3H7-Le) in the toolbox. 

Original [paper](https://openreview.net/pdf?id=Qa3uS3H7-Le) and [code](https://github.com/facebookresearch/ssl-relation-prediction) for Relation Prediction.

Original [paper](https://arxiv.org/abs/2106.12144) and [code](https://github.com/migalkin/NodePiece/tree/main/ogb) for Node Piece.

You can set entity embedder as NodePieceEmbedder to use the trick Node Piece. For example, 

```yaml
import:
- nodepiece_embedder

complex:
  entity_embedder:
    type: nodepiece_embedder
  relation_embedder:
    type: lookup_embedder
    
nodepiece_embedder: 
  dim: 128
  regularize_weight: 0.8e-7
  encoder_dropout: 0.1
  sample_rels: 0
```

See NodePieceEmbedder for more hyperparameters you can use.

LibKGE has an option `wgihts.p` by which you can set the weight of relation prediction loss when you use negative sampling. Compared to the previous loss, this adds p multiplies the loss of relation negative sampling to the total loss, where `weights.p` often be set greater than 0. For example, 

```yaml
negative_sampling:
  weights.p: 0.5
```

In addition, we offered a hyperparameter in 1vsAll so that you can use relation prediction when using 1vsAll. For example, 

```yaml
1vsAll:
  class_name: TrainingJob1vsAll
  relation_prediction: true
  relation_prediction_weight: 0.5
```

Here `relation_prediction_weight` in 1vsAll is equivalent to `weights.p` in negative sampling. 
