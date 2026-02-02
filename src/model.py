import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F


class UnbiasedModel(nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_unit=64, item_num=50000, head_num=1, layer_num=1,
                 max_pos=300, clip_M=0.05, stop_grad_ips=True):
        """
        item_num includes: padding(0), items(1..), EOS(item_num-2), MASK(item_num-1)
        clip_M: clipping threshold M in WangWWW22 Eq.(4)  (w_it = max(p_it, M))
        stop_grad_ips: stop gradient through propensity when optimizing recommender (paper also clips/blocks gradients through IPS)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.hidden_unit = hidden_unit
        self.item_num = item_num
        self.head_num = head_num
        self.layer_num = layer_num
        self.clip_M = float(clip_M)
        self.stop_grad_ips = bool(stop_grad_ips)

        # embeddings
        self.item_emb = nn.Embedding(self.item_num, self.hidden_unit, padding_idx=0)
        self.position_emb = nn.Embedding(max_pos, self.hidden_unit)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Rec encoder stack
        self.net = nn.ModuleList(
            TransformerEncoderLayer(
                d_model=self.hidden_unit,
                nhead=self.head_num,
                dim_feedforward=self.hidden_unit * 4,
                dropout=self.dropout_rate,
                activation="gelu",
            )
            for _ in range(self.layer_num)
        )
        self.recommender = TransformerEncoderLayer(
            d_model=self.hidden_unit,
            nhead=self.head_num,
            dim_feedforward=self.hidden_unit * 4,
            dropout=self.dropout_rate,
            activation="gelu",
        )

        # IPS encoder stack (separate parameters)
        self.ips_net = nn.ModuleList(
            TransformerEncoderLayer(
                d_model=self.hidden_unit,
                nhead=self.head_num,
                dim_feedforward=self.hidden_unit * 4,
                dropout=self.dropout_rate,
                activation="gelu",
            )
            for _ in range(self.layer_num)
        )
        self.ips = TransformerEncoderLayer(
            d_model=self.hidden_unit,
            nhead=self.head_num,
            dim_feedforward=self.hidden_unit * 4,
            dropout=self.dropout_rate,
            activation="gelu",
        )

        self.crossEntropy = nn.CrossEntropyLoss(reduction='none')

    def seq2tensor(self, seqs):
        seqs_emb = self.item_emb(seqs)  # (B,L,D)
        B, L = seqs.shape
        pos = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
        pos = torch.clamp(pos, max=self.position_emb.num_embeddings - 1)
        seqs_emb = seqs_emb + self.position_emb(pos)
        return self.dropout(seqs_emb)

    def encoder(self, seqs, padding_mask):
        x = self.seq2tensor(seqs)              # (B,L,D)
        x = x.transpose(0, 1)                  # (L,B,D)
        for mod in self.net:
            x = mod(x, src_key_padding_mask=padding_mask)
        return x                                # (L,B,D)

    def IPSencoder(self, seqs, padding_mask):
        x = self.seq2tensor(seqs)              # (B,L,D)
        x = x.transpose(0, 1)                  # (L,B,D)
        for mod in self.ips_net:
            x = mod(x, src_key_padding_mask=padding_mask)
        return x                                # (L,B,D)

    # =========================
    # IPS branch: p(v_t | H_{t-1}) estimation (sequence-aware)
    # =========================
    def ips_logits(self, input_seqs):
        """
        return ips logits over all items at each position: (B,L,item_num)
        """
        pad = (input_seqs == 0)
        h = self.IPSencoder(input_seqs, pad)                # (L,B,D)
        h = self.ips(h, src_key_padding_mask=pad)           # (L,B,D)
        logits = torch.matmul(h, self.item_emb.weight.T)    # (L,B,item_num)
        logits = logits.transpose(0, 1)                     # (B,L,item_num)
        return logits

    def ips_pretrain_loss(self, input_seqs):
        """
        Pretrain IPS by maximizing log p(v_t | H_{t-1}) (paper Eq.(20)).
        Here we use teacher forcing on the whole sequence (same target = observed item at each position).
        """
        logits = self.ips_logits(input_seqs)                # (B,L,V)
        logits_t = logits.transpose(1, 2)                   # (B,V,L)
        loss_mat = self.crossEntropy(logits_t, input_seqs)  # (B,L)

        valid = (input_seqs != 0).float()
        loss = (loss_mat * valid).sum() / (valid.sum() + 1e-8)
        return loss

    @torch.no_grad()
    def ips_propensity(self, input_seqs):
        """
        Compute propensity p(v_t | H_{t-1}) for the OBSERVED item at each position.
        Return: p_it (B,L) in (0,1)
        """
        logits = self.ips_logits(input_seqs)                # (B,L,V)
        prob = torch.softmax(logits, dim=-1)                # (B,L,V)
        p_it = prob.gather(dim=-1, index=input_seqs.unsqueeze(-1)).squeeze(-1)  # (B,L)
        return p_it

    # =========================
    # Recommender branch
    # =========================
    def forward(self, input_seqs):
        """
        Recommender logits: (B,L,item_num)
        """
        pad = (input_seqs == 0)
        h = self.encoder(input_seqs, pad)                       # (L,B,D)
        h = self.recommender(h, src_key_padding_mask=pad)       # (L,B,D)
        logits = torch.matmul(h, self.item_emb.weight.T)        # (L,B,V)
        logits = logits.transpose(0, 1)                         # (B,L,V)
        return logits
    
    @torch.no_grad()
    def debias_hard_labels(
        self,
        seq: torch.Tensor,               # (B,L) 原始序列（真值）
        rec_loss_mask: torch.Tensor,     # (B,L) 1 表示监督位置（通常是 mask 的位置）
        mask_token: int = None,          # item_num-1
        bias_q: float = 0.5,             # 每个样本内选多少比例做 bias
        min_k: int = 1,                  # 至少选 1 个（如果该样本有监督位）
    ):
        """
        Hard pseudo labels for bias items:
          - 在每个样本 b 的监督位置集合 S_b 上，
          - 用 masked 输入预测 p(true_item)，取 p 最小的 top-q% 标记为 bias=1.
        return:
          bias_gt: (B,L) float {0,1}
          supervise_mask: (B,L) float {0,1} = rec_loss_mask * nonpad
        """
        if mask_token is None:
            mask_token = self.item_num - 1

        B, L = seq.shape
        pad = (seq == 0)
        supervise_mask = rec_loss_mask.float() * (~pad).float()  # 只在有效 token 位置监督

        # 1) 构造 masked 输入：仅在 supervise_mask==1 的位置 mask
        masked_seq = seq.clone()
        masked_seq[supervise_mask > 0.5] = mask_token

        # 2) 用 recommender forward 得到 logits（注意：输入已 mask，不会泄露真值）
        logits = self.forward(masked_seq)          # (B,L,V)
        probs = torch.softmax(logits, dim=-1)      # (B,L,V)

        # 3) 取真值概率 p_true
        p_true = probs.gather(dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)  # (B,L)

        # 4) 每个样本内：在监督位置里选最小的 top-q%
        bias_gt = torch.zeros_like(p_true, dtype=torch.float32)  # (B,L)

        for b in range(B):
            idx = torch.where(supervise_mask[b] > 0.5)[0]
            if idx.numel() == 0:
                continue
            conf = p_true[b, idx]  # 越小越“可疑”
            k = max(min_k, int(torch.ceil(torch.tensor(bias_q * idx.numel(), device=seq.device)).item()))
            k = min(k, idx.numel())

            # smallest-k
            _, order = torch.topk(conf, k=k, largest=False)
            chosen = idx[order]
            bias_gt[b, chosen] = 1.0

        return bias_gt.long()

    def debiased_recommender_loss(self, rec_logits, target_seqs, rec_loss_mask, ips_p):
        """
        Implements clipped IPS-weighted objective (paper Eq.(4)):
          w_it = max(p_it, M)
          weight = 1 / w_it
          loss = CE * weight * mask
        Notes:
          - ips_p is propensity for OBSERVED item at position t
          - we detach ips_p by default (stop_grad_ips=True)
        """
        # CE over all items
        rec_logits_t = rec_logits.transpose(1, 2)                # (B,V,L)
        ce_mat = self.crossEntropy(rec_logits_t, target_seqs)    # (B,L)

        # IPS weights
        if self.stop_grad_ips:
            ips_p = ips_p.detach()

        w = torch.clamp(ips_p, min=self.clip_M)                  # max(p, M)
        weight = 1.0 / w                                         # 1 / max(p,M)

        # only count masked + non-pad positions
        valid = ((target_seqs != 0).float() * rec_loss_mask.float())
        loss_mat = ce_mat * weight * valid

        denom = valid.sum()
        if denom.item() == 0:
            return loss_mat.sum()
        return loss_mat.sum() / (denom + 1e-8)
    

class model(torch.nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_unit=64, item_num=50000, head_num=1, layer_num=1):
        """
        item_num include total item quantity, padding, EOS token, masked token
        For example, one dataset contains 10000 items
        0-padding 1-10000 item_id 10001-EOS 10002-mask
        EOS is used to determine when sequence insertion stops
        max_insert_size refers to the maximum number of items inserted before each time step
        """
        super(model, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_unit = hidden_unit
        self.item_num = item_num
        self.head_num = head_num
        self.full_layer = nn.Linear(self.hidden_unit, 2)
        self.debiaser_full_layer = nn.Linear(self.hidden_unit, 2)
        self.layer_num = layer_num
        self.net = nn.ModuleList(
            TransformerEncoderLayer(d_model=self.hidden_unit, 
                                    nhead=self.head_num, 
                                    dim_feedforward=self.hidden_unit * 4,
                                    dropout=self.dropout_rate, 
                                    activation="gelu") for _ in range(self.layer_num))
        self.recommender = TransformerEncoderLayer(d_model=self.hidden_unit, 
                                                   nhead=self.head_num, 
                                                   dim_feedforward=self.hidden_unit * 4, 
                                                   dropout=self.dropout_rate, 
                                                   activation="gelu")
        self.item_emb = torch.nn.Embedding(self.item_num, 
                                           self.hidden_unit,
                                           padding_idx=0)
        self.position_emb = torch.nn.Embedding(300, self.hidden_unit)
        self.crossEntropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.dropout = nn.Dropout(self.dropout_rate)

    def seq2tensor(self, seqs):
        """
        Get item embeddings
        """
        seqs_emb = self.item_emb(seqs)
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        positions = torch.tensor(positions, device=seqs_emb.device).long()
        seqs_emb += self.position_emb(positions)

        return seqs_emb

    def encoder(self, seqs, padding_mask):
        """
        Encoder part
        """    
        seqs_emb = self.seq2tensor(seqs)
        seqs_emb = self.dropout(seqs_emb)
        seqs_emb = torch.transpose(seqs_emb, 0, 1)
        encoder_output = seqs_emb
        for mod in self.net:
            encoder_output = mod(encoder_output, src_key_padding_mask=padding_mask)
        return encoder_output
    

    def contrast_loss(self, seq, modified_seq, random_seq, temperature=0.01):
        # encoder: (L, B, D)
        def encode_and_pool(x):
            pad = (x == 0)                      # (B, L) True=PAD
            enc = self.encoder(x, pad)          # (L, B, D)

            keep = (~pad).float()               # (B, L) 1=valid
            keep = keep.transpose(0, 1).unsqueeze(-1)   # (L, B, 1)

            pooled = (enc * keep).sum(dim=0) / (keep.sum(dim=0) + 1e-8)  # (B, D)
            return F.normalize(pooled, dim=-1)

        z = encode_and_pool(seq)               # (B, D)
        z_pos = encode_and_pool(modified_seq)  # (B, D)
        z_neg = encode_and_pool(random_seq)    # (B, D)

        pos_sim = (z * z_pos).sum(dim=-1) / temperature   # (B,)
        neg_sim = (z * z_neg).sum(dim=-1) / temperature   # (B,)

        logits = torch.stack([pos_sim, neg_sim], dim=1)   # (B, 2)
        labels = torch.zeros(seq.size(0), dtype=torch.long, device=seq.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def corrector_loss(self, 
                       full_layer_output, 
                       l1_ground_truth, 
                       padding_mask):
        """
        l1_ground_truth corresponds to the actual operation of each time step (keep, delete or insert)
        That is 0 for keep, 1 for delete, 2 for insert
        l2_ground_truth corresponds to the sequence that should be inserted ahead for each time step
        l1_loss refers to the loss of the keep delete insert operation predicted by the model
        l2_loss calculates the loss doing insert operation
        """
        l1_loss_entropy = self.l1_loss(l1_ground_truth, full_layer_output, padding_mask)

        return l1_loss_entropy


    def l1_loss(self, l1_ground_truth, full_layer_output,
                padding_mask):
        """
        As for the time step of padding, l1_loss equals to 0
        """
        padding_mask = padding_mask.float()  # (batch_size,seqs_len)
        full_layer_output = torch.transpose(full_layer_output, 1, 2)  # (batch_size,2,seqs_len)
        cross_entropy_l1 = self.crossEntropy(full_layer_output, l1_ground_truth)  # (batch_size,seqs_len)
        input_padding = 1 - padding_mask
        cross_entropy_l1 *= input_padding

        return cross_entropy_l1


    def recommender_loss(self, 
                         final_output, 
                         rec_loss_mask, 
                         seqs):
        final_output = torch.transpose(final_output, 1, 2)
        final_loss = self.crossEntropy(final_output, seqs)
        final_loss *= rec_loss_mask
        
        return final_loss
    

    def corrector_forward(self, input_seqs):
        """
        Item-wise corrector part
        :param input_seqs: (batch_size,seqs_len)
        :param input_insert_seqs: (batch_size,seqs_len,max_insert_size-1)
        """
        padding_mask = (input_seqs == 0)
        encoder_output = self.encoder(input_seqs, padding_mask)  # (seqs_len,batch_size,emb_size)
        full_layer_output = self.full_layer(encoder_output)  # (seqs_len,batch_size,2), in addition, 2 refers to keep, delete
        full_layer_output = torch.transpose(full_layer_output, 0, 1)  # (batch_size,seqs_len,2)

        return full_layer_output, padding_mask
    
    
    def debiaser_forward(self, input_seqs):
        """
        Item-wise debiaser part
        :param input_seqs: (batch_size,seqs_len)
        :param input_insert_seqs: (batch_size,seqs_len,max_insert_size-1)
        """
        padding_mask = (input_seqs == 0)
        encoder_output = self.encoder(input_seqs, padding_mask)  # (seqs_len,batch_size,emb_size)
        debiaser_layer_output = self.debiaser_full_layer(encoder_output)  # (seqs_len,batch_size,2), in addition, 2 refers to keep, delete
        debiaser_layer_output = torch.transpose(debiaser_layer_output, 0, 1)  # (batch_size,seqs_len,2)

        return debiaser_layer_output, padding_mask


    def forward(self, input_seqs):
        """
        Recommender part
        """

        padding_mask = (input_seqs == 0)
        encoder_output = self.encoder(input_seqs, padding_mask)  # (seqs_len,batch_size,emb_size)


        recommender_output = self.recommender(encoder_output,
                                              src_key_padding_mask=padding_mask)  
                                              # (seqs_len,batch_size,emb_size)

        recommender_output = torch.matmul(recommender_output, self.item_emb.weight.T)  # seqs_len,batch,item_num
        recommender_output = torch.transpose(recommender_output, 0, 1)  # (batch,seqs_len,item_num)

        return recommender_output
    

    def corrector_inference(self, input_seqs, return_prob: bool = True):
        padding_mask = (input_seqs == 0)
        encoder_output = self.encoder(input_seqs, padding_mask)
        logits = self.full_layer(encoder_output)      # (L,B,C) or (B,L,C) 取决于你的实现
        # 你现在 argmax(-1) 后又 transpose，说明 logits 目前是 (L,B,C)
        prob = torch.softmax(logits, dim=-1)          # (L,B,C)
        decisions = prob.argmax(dim=-1).transpose(0,1) # (B,L)

        if not return_prob:
            return decisions
        prob = prob.transpose(0,1)                    # (B,L,C)
        return decisions
    
    def debiaser_inference(self, input_seqs):
        """
        Debias the original sequence by the debiaser model
        return: sequence after debiasing the original sequence
        """
        #
        padding_mask = (input_seqs == 0)
        encoder_output = self.encoder(input_seqs, padding_mask)
        debiaser_layer_output = self.debiaser_full_layer(encoder_output)
        prob = torch.softmax(debiaser_layer_output, dim=-1)
        
        decisions = debiaser_layer_output.argmax(-1)  # (seqs_len, batch_size)
        decisions = torch.transpose(decisions, 0, 1)  # (batch_size, seqs_len)
        prob = prob.transpose(0, 1)

        return decisions

    def seq_correction(self, decisions, input_seqs):
        # Apply the delete operation to the sequence only
        modified_seqs = input_seqs.clone()
        decisions[modified_seqs == 0] = 0
        modified_seqs[decisions == 1] = 0
        modified_seqs = modified_seqs.tolist()

        batch_size = len(modified_seqs)
        for i in range(batch_size):
            modified_seqs[i] = list(filter(lambda x: (x != 0 and x != self.item_num - 2 and x != self.item_num - 1), modified_seqs[i]))   # filter padding, EOS, mask token

        return modified_seqs



class SeqMerger(nn.Module):
    def __init__(self, item_num, hidden=64, head_num=2, layer_num=2, max_pos=300, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(item_num, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_pos, hidden)
        self.dropout  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden,
                nhead=head_num,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
            ) for _ in range(layer_num)
        ])
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_seqs):
        pad = (input_seqs == 0)
        B, L = input_seqs.shape
        pos = torch.arange(L, device=input_seqs.device).unsqueeze(0).expand(B, L)
        pos = torch.clamp(pos, max=self.pos_emb.num_embeddings - 1)

        x = self.item_emb(input_seqs) + self.pos_emb(pos)
        x = self.dropout(x)          # (B,L,D)
        x = x.transpose(0, 1)        # (L,B,D)

        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=pad)

        x = x.transpose(0, 1)        # (B,L,D)
        gate_logits = self.head(x).squeeze(-1)  # (B,L)
        if self.training:
            gate_logits = gate_logits + 0.01 * torch.randn_like(gate_logits)
        return gate_logits   
