import torch
from kgbench import Config, Dataset
from kgbench.model.kge_model import RelationalScorer, KgeModel
import numpy as np

class AutoBLMScorer(RelationalScorer):
    r"""Implementation of the AutoBLM KGE scorer.

    Reference: Zhang, Yongqi and Yao, Quanming and Kwok, T James: 
    Bilinear Scoring Function Search for Knowledge Graph Learning. 
    In TPAMI 2022. 
    `<https://ieeexplore.ieee.org/document/9729658>`

    """
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.A = self.get_option("A")
        self.K = self.get_option("K")
        self.A_np = np.array(self.A)
        self.A_np_notzero = np.flatnonzero(self.A_np)


    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        s_emb_chunk = s_emb.chunk(self.K, dim=1)
        p_emb_chunk = p_emb.chunk(self.K, dim=1)
        o_emb_chunk = o_emb.chunk(self.K, dim=1)

        if combine == "spo":
            score = torch.zeros(n).to(self.config.get("job.device"))
            for idx in self.A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = self.A_np[idx]
                score += np.sign(A_ij) * (s_emb_chunk[ii] * p_emb_chunk[abs(A_ij)-1] * o_emb_chunk[jj]).sum(dim=1)

        elif combine == "sp_":
            mult_part = torch.zeros(n, p_emb.size(1)).view(n, self.K, -1).to(self.config.get("job.device"))
            for idx in self.A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = self.A_np[idx]
                mult_part[:,jj] += np.sign(A_ij) * (s_emb_chunk[ii] * p_emb_chunk[abs(A_ij)-1])
            mult_part = mult_part.view(n, -1)
            score = mult_part.mm(o_emb.transpose(0,1))

        elif combine == "_po":
            mult_part = torch.zeros(n, p_emb.size(1)).view(n, self.K, -1).to(self.config.get("job.device"))
            for idx in self.A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = self.A_np[idx]
                mult_part[:,ii] += np.sign(A_ij) * (o_emb_chunk[jj] * p_emb_chunk[abs(A_ij)-1])
            mult_part = mult_part.view(n, -1)
            score = mult_part.mm(s_emb.transpose(0,1))

        elif combine == "s_o":
            n = s_emb.size(0)
            mult_part = torch.zeros(n, s_emb.size(1)).view(n, self.K, -1).to(self.config.get("job.device"))
            for idx in self.A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = self.A_np[idx]
                mult_part[:,abs(A_ij)-1] += np.sign(A_ij) * (s_emb_chunk[ii] * o_emb_chunk[jj])
            mult_part = mult_part.view(n, -1)
            score = mult_part.mm(p_emb.transpose(0,1))

        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return score.view(n, -1)      


    def score_emb_sp_given_negs(self, s_emb: torch.Tensor , p_emb: torch.Tensor, o_emb: torch.Tensor):
        r""" 
            Compute scores for hr_ queries with given neg tails, used for ogbl dataset eval.
            s_emb: batch_size * dim
            p_emb: batch_size * dim
            o_emb: batch_size * negs * dim
         """
        
        n = p_emb.size(0)

        s_emb_chunk = s_emb.chunk(self.K, dim=-1)
        p_emb_chunk = p_emb.chunk(self.K, dim=-1)
        
        A_np = np.array(self.A)
        A_np_notzero = np.flatnonzero(A_np)


        mult_part = torch.zeros(n, p_emb.size(1)).view(n, self.K, -1).to(self.config.get("job.device"))
        for idx in A_np_notzero:
            ii = idx // self.K  # row: s
            jj = idx % self.K   # column: o
            A_ij = A_np[idx]
            mult_part[:,jj] += np.sign(A_ij) * (s_emb_chunk[ii] * p_emb_chunk[abs(A_ij)-1])
        mult_part = mult_part.view(n, -1)

        return (mult_part.unsqueeze(dim=2) * (o_emb.transpose(1,2))).sum(dim=1)

    def score_emb_po_given_negs(self, s_emb: torch.Tensor , p_emb: torch.Tensor, o_emb: torch.Tensor):
        r""" 
            Compute scores for _rt queries with given neg heads, used for ogbl dataset eval.
            s_emb: batch_size * negs * dim
            p_emb: batch_size * dim
            o_emb: batch_size * dim
         """

        n = p_emb.size(0)

        o_emb_chunk = o_emb.chunk(self.K, dim=-1)
        p_emb_chunk = p_emb.chunk(self.K, dim=-1)
        
        A_np = np.array(self.A)
        A_np_notzero = np.flatnonzero(A_np)

        mult_part = torch.zeros(n, p_emb.size(1)).view(n, self.K, -1).to(self.config.get("job.device"))
        for idx in A_np_notzero:
            ii = idx // self.K  # row: s
            jj = idx % self.K   # column: o
            A_ij = A_np[idx]
            mult_part[:,ii] += np.sign(A_ij) * (o_emb_chunk[jj] * p_emb_chunk[abs(A_ij)-1])
        mult_part = mult_part.view(n, -1)

        return (mult_part.unsqueeze(dim=2) * (s_emb.transpose(1,2))).sum(dim=1)



class AutoBLM(KgeModel):
    r"""Implementation of the AutoBLM KGE model."""
    
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=AutoBLMScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )





