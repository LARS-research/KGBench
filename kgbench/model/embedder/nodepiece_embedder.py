from torch import Tensor
import torch.nn
import torch.nn.functional

from kgbench import Config, Dataset
from kgbench.job import Job
from kgbench.model import KgeEmbedder
from kgbench.misc import round_to_points

from typing import List

import torch
import numpy as np
import pickle
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import random

from collections import defaultdict


class NodePieceEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.regularize = self.check_option("regularize", ["", "lp"])
        # self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size
        self.use_distances = self.get_option('use_dists')
        self.sample_paths = self.get_option('max_paths')
        self.sample_rels = self.get_option('sample_rels')
        self.pooler = self.get_option('pooler')
        self.NOTHING_TOKEN = -99
        self.CLS_TOKEN = -1
        self.MASK_TOKEN = -10
        self.PADDING_TOKEN = -100
        self.SEP_TOKEN = -2

        self.top_entities, self.other_entities, self.vocab = self.tokenize_kg()

        self.token2id = {t: i for i, t in enumerate(self.top_entities)}
        self.token2id[self.NOTHING_TOKEN] = len(self.token2id)
        # self.rel2token = {t: i + len(self.top_entities) for i, t in
        #                   enumerate(list(self.triples_factory.relation_to_id.values()))}
        self.nentity = self.config.get('dataset.num_entities')
        self.nrelation = self.config.get('dataset.num_relations')
        # self.vocab_size = len(self.token2id) + self.nrelation * (1
        #                     if self.config.get('model')!='reciprocal_relations_model' else 2)

        self.anchor_embeddings = torch.nn.Embedding(num_embeddings=len(self.token2id)+1, embedding_dim=self.dim)
        hashes = [
            [self.token2id[token] for token in vals['ancs'][:min(self.sample_paths, len(vals['ancs']))]] + [
                self.token2id[self.PADDING_TOKEN]] * (self.sample_paths - len(vals['ancs']))
            for entity, vals in self.vocab.items()
        ]
        distances = [
            [d for d in vals['dists'][:min(self.sample_paths, len(vals['dists']))]] + [0] * (
                        self.sample_paths - len(vals['dists']))
            for entity, vals in self.vocab.items()
        ]

        self.max_seq_len = max([d for row in distances for d in row])
        print(
            f"Max seq len is {self.max_seq_len}. Keep {self.sample_paths} shortest paths")

        self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.config.get("job.device"))
        self.distances = torch.tensor(distances, dtype=torch.long, device=self.config.get("job.device"))

        print("Creating relational context")
        if self.sample_rels > 0:
            pad_idx = self.nrelation
            e2r = defaultdict(set)
            for i in tqdm(range(self.dataset._triples['train'].shape[0])):
                e2r[self.dataset._triples['train'][i,0].item()].add(self.dataset._triples['train'][i,1].item())

            len_stats = [len(v) for k,v in e2r.items()]
            print(f"Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(len_stats, 66)}, max: {max(len_stats)} ")
            unique_1hop_relations = [
                random.sample(e2r[i], k=min(self.sample_rels, len(e2r[i]))) + [pad_idx] * (self.sample_rels-min(len(e2r[i]), self.sample_rels))
                for i in range(self.nentity)
            ]
            self.unique_1hop_relations = torch.tensor(unique_1hop_relations, dtype=torch.long, device=self.config.get("job.device"))


        # distance integer to denote path lengths
        self.dist_embeddings = torch.nn.Embedding(self.max_seq_len + 1, embedding_dim=self.dim)
        self.relcontext_embeddings = torch.nn.Embedding(num_embeddings=self.nrelation + 1, embedding_dim=self.dim)
        
        if not init_for_load_only:
            # initialize weights
            self.initialize(self.anchor_embeddings.weight.data)
            self.initialize(self.dist_embeddings.weight.data)
            self.initialize(self.relcontext_embeddings.weight.data)
            self._normalize_embeddings()

        with torch.no_grad():
            self.anchor_embeddings.weight[self.token2id[self.PADDING_TOKEN]] = torch.zeros(self.dim)
            self.relcontext_embeddings.weight.data[-1] = torch.zeros(self.dim)
            self.dist_embeddings.weight[0] = torch.zeros(self.dim)

        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)
        
        self.encoder_dropout = self.get_option('encoder_dropout')
        # print("self.dim * (self.sample_paths + self.sample_rels)", self.dim * (self.sample_paths + self.sample_rels))
        # input()
        self.set_enc = torch.nn.Sequential(
            torch.nn.Linear(self.dim * (self.sample_paths + self.sample_rels), self.dim * 2), torch.nn.Dropout(self.encoder_dropout), torch.nn.ReLU(),
            torch.nn.Linear(self.dim * 2, self.dim))
        # print("self.set_enc", self.set_enc)
        # init
        for module in self.set_enc.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()



    def _normalize_embeddings(self):
        if self.normalize_p > 0:
            with torch.no_grad():
                self.anchor_embeddings.weight.data = torch.nn.functional.normalize(
                    self.anchor_embeddings.weight.data, p=self.normalize_p, dim=-1
                )
                self.dist_embeddings.weight.data = torch.nn.functional.normalize(
                    self.dist_embeddings.weight.data, p=self.normalize_p, dim=-1
                )
                self.relcontext_embeddings.weight.data = torch.nn.functional.normalize(
                    self.relcontext_embeddings.weight.data, p=self.normalize_p, dim=-1
                )

    def prepare_job(self, job: Job, **kwargs):
        from kgbench.job import TrainingJob

        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0 and isinstance(job, TrainingJob):
            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self._normalize_embeddings())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self._normalize_embeddings())

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: KgeEmbedder) -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        for embedding in [self.anchor_embeddings, self.dist_embeddings, self.relcontext_embeddings]:
            embedding.weight[
                torch.from_numpy(self_intersect_ind)
                .to(embedding.weight.device)
                .long()
            ] = pretrained_embedder.embed(torch.from_numpy(pretrained_intersect_ind)).to(
                embedding.weight.device
            )

    def pool_anchors(self, anc_embs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        input shape: (bs, num_anchors, emb_dim)
        output shape: (bs, emb_dim)
        """


        # print("anc_embs.shape", anc_embs.shape)
        # print("nan,inf", anc_embs.isnan().any(), anc_embs.isinf().any())

        if self.pooler == "set":
            pooled = self.set_enc(anc_embs)
        elif self.pooler == "cat":
            if len(anc_embs.shape) == 3:
                anc_embs = anc_embs.view(anc_embs.shape[0], -1)
                pooled = self.set_enc(anc_embs) if self.sample_paths != 1 else anc_embs
            else:
                s0 = anc_embs.shape[0]
                s1 = anc_embs.shape[1]
                anc_embs = anc_embs.view(s0*s1, -1)
                pooled = self.set_enc(anc_embs) if self.sample_paths != 1 else anc_embs
                pooled = pooled.view(s0, s1, -1)
            # print("anc_embs.shape", anc_embs.shape)
            # a = torch.rand((2048,8192)).to('cuda:0')
            # pooled = self.set_enc(a)
            # print("1111111111111")
            # pooled = self.set_enc(anc_embs)
            # print("000000000000")
            
        elif self.pooler == "trf" or self.pooler == "moe":
            pooled = self.set_enc(anc_embs.transpose(1, 0))  # output shape: (seq_len, bs, dim)
            pooled = pooled.mean(dim=0)  # output shape: (bs, dim)
            if self.policy == "cat":
                pooled = self.linear(pooled)

        return pooled

    def embed(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self.embedder(indexes.long()))

    def embedder(self, indexes: Tensor) -> Tensor:
        hashes, dists = self.hashes[indexes], self.distances[indexes]

        #anc_embs = torch.index_select(self.anchor_embeddings, dim=0, index=hashes)
        anc_embs = self.anchor_embeddings(hashes)
        mask = None

        # print("max", hashes.max(), dists.max(), indexes.max())
        # print("min", hashes.min(), dists.min(), indexes.min())

        if self.use_distances:
            dist_embs = self.dist_embeddings(dists)
            anc_embs += dist_embs

        if self.sample_rels > 0:
            rels = self.unique_1hop_relations[indexes]  # (bs, rel_sample_size)
            #rels = torch.index_select(self.relation_embedding, dim=0, index=rels)   # (bs, rel_sample_size, dim)
            # print("max min", rels.max(), rels.min())
            
            rels = self.relcontext_embeddings(rels)
            anc_embs = torch.cat([anc_embs, rels], dim=-2)  # (bs, [num_negs, ]ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask)
        return anc_embs

    def embed_all(self) -> Tensor:
        return self._postprocess(self._embeddings_all())

    def _embeddings_anchors(self) -> Tensor:
        return torch.cat(
            (self.anchor_embeddings(
                torch.arange(len(self.token2id)+1, dtype=torch.long, device=self.config.get("job.device"))),
            self.dist_embeddings(
                torch.arange(self.max_seq_len + 1, dtype=torch.long, device=self.config.get("job.device"))),
            self.relcontext_embeddings(
                torch.arange(self.nrelation + 1, dtype=torch.long, device=self.config.get("job.device")))
            ), dim = 0)

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> Tensor:
        embed_bsz = 500_000
        if self.vocab_size < embed_bsz:
            return self.embedder(torch.arange(self.vocab_size, dtype=torch.long, device=self.config.get("job.device")))
        else:
            embed = torch.empty((0, self.dim)).to(self.config.get("job.device"))
            for i in range(((self.vocab_size-1)//embed_bsz) + 1):
                embed = torch.cat((embed,self.embedder(torch.arange((i-1)*embed_bsz, min(i*embed_bsz,self.vocab_size), dtype=torch.long, device=self.config.get("job.device")))), dim=0)
            return embed

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_anchors()
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self.embedder(unique_indexes)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result

    def tokenize_kg(self):

        self.tokenizer = self.get_option('tokenizer')
        self.anchor_strategy = self.tokenizer['anchor_strategy']
        strategy_encoding = f"d{self.anchor_strategy['degree']}_b{self.anchor_strategy['betweenness']}_p{self.anchor_strategy['pagerank']}_r{self.anchor_strategy['random']}"

        filename = f"data/{self.config.get('dataset.name')}/{self.config.get('dataset.name')}_{self.tokenizer['num_anchors']}_anchors_{self.tokenizer['num_paths']}_paths_{strategy_encoding}_pykeen"
        if self.tokenizer['sp_limit'] > 0:
            filename += f"_{self.tokenizer['sp_limit']}sp"  # for separating vocabs with limited mined shortest paths
        if self.tokenizer['rand_limit'] > 0:
            filename += f"_{self.tokenizer['sp_limit']}rand"
        if self.tokenizer['tkn_mode'] == "bfs":
            filename += "_bfs"
        if self.tokenizer['partition'] > 1:
            filename += f"_metis{self.tokenizer['partition']}"
        filename += ".pkl"
        path = Path(filename)
        if path.is_file():
            anchors, non_anchors, vocab = pickle.load(open(path, "rb"))
            return anchors, non_anchors, vocab
        else:
            raise ValueError(f"Cannot find token file {filename}")


