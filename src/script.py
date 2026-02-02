import random
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
import time
import os
import json


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def xavier_init(model):
    for name, par in model.named_parameters():
        if 'weight' in name and len(par.shape) >= 2:
            nn.init.xavier_normal_(par)
        elif 'bias' in name:
            nn.init.constant_(par, 0.0)


def seqs_normalization_test(modified_leave_one_seqs, modified_max_seqs_len, item_num):
    """
    :param item_num
    :param modified_max_seqs_len
    :param modified_leave_one_seqs: list[list]
    :return: (batch_size,seqs_len)
    """
    modified_masked_seqs = []
    for k, v in enumerate(modified_leave_one_seqs):
        # if sequence length exceeds (modified_max_sequence_len-1), delete the previous part of the sequence
        if len(v) > modified_max_seqs_len - 1:
            masked_seqs_insert = modified_leave_one_seqs[k][1 - modified_max_seqs_len:]
            masked_seqs_insert.append(item_num - 1)
            modified_masked_seqs.append(masked_seqs_insert)
        elif len(v) <= modified_max_seqs_len-1:
            masked_seqs_insert = modified_leave_one_seqs[k].copy()
            masked_seqs_insert.append(item_num - 1)
            masked_seqs_insert += [0] * (modified_max_seqs_len-len(masked_seqs_insert))
            modified_masked_seqs.append(masked_seqs_insert)
    modified_masked_seqs = torch.tensor(modified_masked_seqs, dtype=torch.long)
    return modified_masked_seqs


def seqs_normalization(modified_seqs, modified_max_seqs_len, item_num, mask_prob):
    """
    :param item_num
    :param modified_max_seqs_len
    :param modified_seqs: list[list]
    :param mask_prob
    :return: tensor(batch_size,seqs_len)
    """
    modified_masked_seqs = torch.zeros([len(modified_seqs), modified_max_seqs_len], dtype=torch.long)
    modified_rec_loss_mask = torch.zeros([len(modified_seqs), modified_max_seqs_len], dtype=torch.float)
    modified_ori_seqs = torch.zeros([len(modified_seqs), modified_max_seqs_len], dtype=torch.long)
    for k, v in enumerate(modified_seqs):
        # If sequence length exceeds (modified_max_sequence_len-1), delete the previous part of the sequence
        if len(v) > modified_max_seqs_len:
            temp_modified_seqs = v[- modified_max_seqs_len:]
            temp_modified_seqs = torch.tensor(temp_modified_seqs, dtype=torch.long)
            modified_ori_seqs[k] = temp_modified_seqs
            modified_rec_loss_mask[k][np.random.rand(modified_max_seqs_len) < mask_prob] = 1
            temp_modified_seqs[modified_rec_loss_mask[k] == 1] = item_num - 1
            modified_masked_seqs[k] = temp_modified_seqs
        elif modified_max_seqs_len >= len(v) > 0:
            padding_mask = torch.zeros([modified_max_seqs_len], dtype=torch.float)
            padding_mask[:len(v)] = 1
            temp_modified_seqs = v + [0] * (modified_max_seqs_len - len(v))
            temp_modified_seqs = torch.tensor(temp_modified_seqs, dtype=torch.long)
            modified_ori_seqs[k] = temp_modified_seqs
            modified_rec_loss_mask[k][np.random.rand(modified_max_seqs_len) < mask_prob] = 1
            modified_rec_loss_mask[k] = modified_rec_loss_mask[k]*padding_mask
            temp_modified_seqs[modified_rec_loss_mask[k] == 1] = item_num - 1
            modified_masked_seqs[k] = temp_modified_seqs
        # If the sequence length is 0, set the first time step to EOS so as to avoid nan error
        else:
            temp_modified_seqs = v + [item_num - 2] + [0] * (modified_max_seqs_len - 1)  # avoid the nan error
            temp_modified_seqs = torch.tensor(temp_modified_seqs, dtype=torch.long)             
            modified_masked_seqs[k] = temp_modified_seqs
    return modified_masked_seqs, modified_rec_loss_mask, modified_ori_seqs



def evaluate_function(output, positives, negatives):
    result = [{} for _ in range(len(output))]
    for i in range(len(output)):
        pos_score = output[i][positives[i]]
        neg_scores = output[i][negatives[i]]
        success = ((neg_scores - pos_score) < 0).sum()
        success = int(success)
        rank = len(negatives[i]) - success + 1
        for k in (5, 10, 20):
            key_r = f'recall@{k}'
            key_m = f'mrr@{k}'
            key_n = f'ndcg@{k}'
            if rank <= k:
                result[i][key_r] = 1.0
                result[i][key_m] = 1.0 / rank
                result[i][key_n] = 1.0 / np.log2(rank + 1)
            else:
                result[i][key_r] = 0.0
                result[i][key_m] = 0.0
                result[i][key_n] = 0.0

    return result


def get_metrics(metrics_name,total_result):
    if metrics_name == 'recall@5':
        recall5 = 0.0
        for i in total_result:
            recall5 += i['recall@5']
        return recall5 / len(total_result)
    elif metrics_name == 'mrr@5':
        mrr5 = 0.0
        for i in total_result:
            mrr5 += i['mrr@5']
        return mrr5 / len(total_result)
    elif metrics_name == 'recall@10':
        recall10 = 0.0
        for i in total_result:
            recall10 += i['recall@10']
        return recall10 / len(total_result)
    elif metrics_name == 'mrr@10':
        mrr10 = 0.0
        for i in total_result:
            mrr10 += i['mrr@10']
        return mrr10 / len(total_result)
    elif metrics_name == 'recall@20':
        recall20 = 0.0
        for i in total_result:
            recall20 += i['recall@20']
        return recall20 / len(total_result)
    elif metrics_name == 'mrr@20':
        mrr20 = 0.0
        for i in total_result:
            mrr20 += i['mrr@20']
        return mrr20 / len(total_result)
    elif metrics_name == 'ndcg@5':
        ndcg5 = 0.0
        for i in total_result:
            ndcg5 += i['ndcg@5']
        return ndcg5 / len(total_result)
    elif metrics_name == 'ndcg@10':
        ndcg5 = 0.0
        for i in total_result:
            ndcg5 += i['ndcg@10']
        return ndcg5 / len(total_result)
    elif metrics_name == 'ndcg@20':
        ndcg5 = 0.0
        for i in total_result:
            ndcg5 += i['ndcg@20']
        return ndcg5 / len(total_result)
    else:
        raise Exception("error!")



def save_metrics(args, name: str, metrics: dict):
    out_dir = os.path.join(args.o, "results")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{name}.json")
    payload = {
        "dataset": args.dataset,
        "seed": getattr(args, "seed", None),
        "eval_mode": getattr(args, "eval_mode", None),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {path}")
