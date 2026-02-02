import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
import copy

import pickle

from model import model, UnbiasedModel, SeqMerger
from dataLoader import TrainDataset, ValidDataset, TestDataset
from script import (
    init_seeds, xavier_init,
    seqs_normalization, seqs_normalization_test,
    evaluate_function, get_metrics,
    save_metrics
)

import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)

# =========================
# parser (keep your legacy args)
# =========================
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ml1m', help='dataset name.')

    # legacy path args
    parser.add_argument('-tf', type=str, default='', help='path of training dataset')
    parser.add_argument('-vf', type=str, default='', help='path of valid dataset')
    parser.add_argument('-ef', type=str, default='', help='path of test dataset')
    parser.add_argument('-vn', type=str, default='', help='path of valid_neg dataset')
    parser.add_argument('-en', type=str, default='', help='path of test_neg dataset')

    # training
    parser.add_argument('-b',  type=int,   default=512,   help='batch_size')
    parser.add_argument('-ls', type=int,   default=1000,  help='log_step')
    parser.add_argument('-l',  type=float, default=1e-3,  help='learning_rate')
    parser.add_argument('-e',  type=int,   default=1000,  help='epoch_num')
    parser.add_argument('-dr', type=float, default=0.2,   help='dropout_rate')
    parser.add_argument('-hd', type=int,   default=128,   help='hidden_dim')
    parser.add_argument('-hn', type=int,   default=2,     help='head_num')
    parser.add_argument('-ln', type=int,   default=2,     help='transformer_layer_num')
    parser.add_argument('-o',  type=str,   default='./save_model/', help='base save_path')
    parser.add_argument('-r',  action='store_true', help='resume (load best.pth)')
    parser.add_argument('-seed', type=int, default=961, help='random seed')

    # item num / seq lens
    parser.add_argument('-n',  type=int, default=3419, help='item_num (incl. PAD/EOS/MASK)') # ml-1m
    parser.add_argument('-ml', type=int, default=200,  help='max_seqs_len')
    parser.add_argument('-mml',type=int, default=50,   help='modified_max_seqs_len')
    parser.add_argument('-mi', type=int, default=1,    help='max_insert_num')
    parser.add_argument('-mb', type=float, default=0.5, help='mask_prob')
    parser.add_argument('-p',  type=float, default=0.4, help='<=p keep else delete')
    parser.add_argument('--rho', type=float, default=0.5, help='bias prob.')

    # validation control
    parser.add_argument('--val_freq', type=int, default=10, help='validate every N epochs')
    parser.add_argument('--patience', type=int, default=3,  help='early stop if no improvement for K validations')

    # evaluation mode
    parser.add_argument('--eval_mode', type=str, default='orig',
                        choices=['orig','del','ins','both','all'],
                        help="orig|del|ins|both|all")

    parser.add_argument('--test_only', type=bool, default=False,
                        help='only run test with existing ckpts')

    # IPS teacher
    parser.add_argument('--clip_M', type=float, default=0.05, help='IPS clipping M')
    parser.add_argument('--ips_pre_epochs', type=int, default=10, help='max IPS pretrain epochs')

    # merger hyperparams
    parser.add_argument('--merger_hidden', type=int, default=128)
    parser.add_argument('--merger_hn', type=int, default=2)
    parser.add_argument('--merger_ln', type=int, default=2)
    parser.add_argument('--merger_dropout', type=float, default=0.5)

    # optional merger train controls (safe defaults)
    parser.add_argument('--merger_lr', type=float, default=1e-5)
    parser.add_argument('--merger_epochs', type=int, default=1000)
    parser.add_argument('--merger_val_freq', type=int, default=10)
    parser.add_argument('--merger_patience', type=int, default=3)

    return parser


def init_paths(args):
    ds = args.dataset
    base_dir = f'../data/{ds}'
    args.tf = args.tf or os.path.join(base_dir, 'train.dat')
    args.vf = args.vf or os.path.join(base_dir, 'valid.dat')
    args.ef = args.ef or os.path.join(base_dir, 'test.dat')
    args.vn = args.vn or os.path.join(base_dir, 'valid_neg.dat')
    args.en = args.en or os.path.join(base_dir, 'test_neg.dat')

    return args


@torch.no_grad()
def validate(cur_model, dataloader, item_num, modified_max_seqs_len, device, eval_mode='both'):
    cur_model.eval()

    def eval_one_mode(mode, batch):
        seq, neg, masked_seq, leave_one_seq, target = batch
        neg = neg.to(device, non_blocking=True)
        masked_seq = masked_seq.to(device, non_blocking=True)
        leave_one_seq = leave_one_seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if mode == 'orig':
            out = cur_model.forward(masked_seq)
            out = out[masked_seq == item_num - 1]
            return evaluate_function(out, target, neg), [], []

        decisions_ori = cur_model.corrector_inference(leave_one_seq)
        mod_seqs = cur_model.seq_correction(decisions_ori, leave_one_seq)
        mod_masked = seqs_normalization_test(mod_seqs, modified_max_seqs_len, item_num).to(device, non_blocking=True)
        out = cur_model.forward(mod_masked)
        out = out[mod_masked == item_num - 1]

        return evaluate_function(out, target, neg), [], []

    all_modes = ['orig','del','both'] if eval_mode == 'all' else [eval_mode]

    result_dict = {m: [] for m in all_modes}
    sim_dict    = {m: [] for m in all_modes}
    op_dict     = {m: [] for m in all_modes}

    for batch in dataloader:
        for m in all_modes:
            res, sim, op = eval_one_mode(m, batch)
            result_dict[m].extend(res)
            sim_dict[m].extend(sim)
            op_dict[m].extend(op)

    reports = {}
    for m in all_modes:
        res = result_dict[m]
        metrics = {
            'recall@5' : get_metrics('recall@5',  res),
            'recall@10': get_metrics('recall@10', res),
            'recall@20': get_metrics('recall@20', res),
            'ndcg@5': get_metrics('ndcg@5', res),
            'ndcg@10': get_metrics('ndcg@10', res),
            'ndcg@20': get_metrics('ndcg@20', res),
            'mrr@5'    : get_metrics('mrr@5',     res),
            'mrr@10'   : get_metrics('mrr@10',    res),
            'mrr@20'   : get_metrics('mrr@20',    res),
        }
        metrics['sum'] = metrics['recall@5'] + metrics['recall@10'] + metrics['recall@20'] + \
                         metrics['mrr@5'] + metrics['mrr@10'] + metrics['mrr@20'] + \
                         metrics['ndcg@5'] + metrics['ndcg@10'] + metrics['ndcg@20']

        if len(op_dict[m]) > 0:
            arr = np.array(op_dict[m])
            arr = arr.sum(0) / arr.sum()
            reports[m] = {
                'metrics': metrics,
                'similarity': sum(sim_dict[m]) / max(1, len(sim_dict[m])),
                'operation': arr.tolist()
            }
        else:
            reports[m] = {'metrics': metrics}

    key_metric = reports['orig']['metrics']['ndcg@10']
    return reports, key_metric


@torch.no_grad()
def validate_unbiased_recommender(net, dataloader, item_num, device):
    net.eval()
    total_result = []

    for batch in dataloader:
        seq, neg, masked_seq, leave_one_seq, target = batch
        neg = neg.to(device, non_blocking=True)
        masked_seq = masked_seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = net.forward(masked_seq)
        out = logits[masked_seq == item_num - 1]
        total_result.extend(evaluate_function(out, target, neg))

    metrics = {
        'recall@5' : get_metrics('recall@5',  total_result),
        'recall@10': get_metrics('recall@10', total_result),
        'recall@20': get_metrics('recall@20', total_result),
        'mrr@5'    : get_metrics('mrr@5',     total_result),
        'mrr@10'   : get_metrics('mrr@10',    total_result),
        'mrr@20'   : get_metrics('mrr@20',    total_result),
        'ndcg@5': get_metrics('ndcg@5', total_result),
        'ndcg@10': get_metrics('ndcg@10', total_result),
        'ndcg@20': get_metrics('ndcg@20', total_result),
    }
    metrics['sum'] = metrics['recall@5'] + metrics['recall@10'] + metrics['recall@20'] + \
                     metrics['mrr@5'] + metrics['mrr@10'] + metrics['mrr@20'] + \
                     metrics['ndcg@5'] + metrics['ndcg@10'] + metrics['ndcg@20']
    return metrics, metrics['ndcg@10']


def train_unbiased_recommender(args):
    init_seeds(seed=args.seed)

    os.makedirs(args.o, exist_ok=True)
    model_dir = os.path.join(args.o, 'model')
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, 'unbiased_best.pth')

    train_dataset = TrainDataset(args.tf, args.n, args.ml, args.mi, args.mb, args.p)
    valid_dataset = ValidDataset(args.vf, args.vn, args.n, args.mml)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.b, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UnbiasedModel(args.dr, args.hd, args.n, args.hn, args.ln, clip_M=args.clip_M).to(device)

    net.apply(xavier_init)

    optimizer = optim.Adam(net.parameters(), lr=args.l)

    # Stage A: IPS pretrain (train-loss early stop)
    print('==== Stage A: IPS pretrain ====')
    best_dummy = 1e18
    bad = 0
    for epoch in range(1, args.ips_pre_epochs + 1):
        net.train()
        tic = time.time()
        acc = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            seq, masked_seq, *_ = batch
            seq = seq.to(device, non_blocking=True)

            loss = net.ips_pretrain_loss(seq)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            acc += float(loss.detach().cpu().item())
            if step % args.ls == 0:
                print(f'[IPS] epoch {epoch} step {step} loss {acc/step:.4f} time {int(time.time()-tic)}s')

        epoch_loss = acc / max(1, step)
        print(f'[IPS] epoch {epoch} avg_loss {epoch_loss:.6f}')
        if epoch_loss < best_dummy:
            best_dummy = epoch_loss
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f'[IPS] early stop @ epoch {epoch}')
                break

    # Stage B: debiased recommender training (validate)
    print('==== Stage B: Debiased recommender training ====')
    best_metric = -1e18
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.e + 1):
        net.train()
        tic = time.time()
        acc = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            seq, masked_seq, *_rest, rec_loss_mask = batch
            seq = seq.to(device, non_blocking=True)
            masked_seq = masked_seq.to(device, non_blocking=True)
            rec_loss_mask = rec_loss_mask.to(device, non_blocking=True)

            rec_logits = net.forward(masked_seq)            # (B,L,V)
            ips_p = net.ips_propensity(seq)                 # (B,L)
            loss = net.debiased_recommender_loss(rec_logits, seq, rec_loss_mask, ips_p)

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            acc += float(loss.detach().cpu().item())
            if step % args.ls == 0:
                print(f'[REC] epoch {epoch} step {step} loss {acc/step:.4f} time {int(time.time()-tic)}s')

        print(f'[REC] epoch {epoch} avg_loss {acc/max(1,step):.6f}')

        if epoch % args.val_freq == 0:
            metrics, key = validate_unbiased_recommender(net, valid_loader, args.n, device)
            print(f'[VALID] epoch {epoch} ndcg@10={key:.6f} metrics={metrics}')

            if key > best_metric:
                best_metric = key
                best_epoch = epoch
                bad = 0
                torch.save(net.state_dict(), best_path)
                print(f'>>> New best saved @ epoch {epoch}, ndcg@10={best_metric:.6f}')
            else:
                bad += 1
                print(f'No improvement for {bad} validation(s).')
                if bad >= args.patience:
                    print(f'*** Early stopping @ epoch {epoch}. Best @ epoch {best_epoch}, ndcg@10={best_metric:.6f}')
                    break

    return best_path


def train_with_denoise(args, model_name='best_denoise.pth'):
    init_seeds(seed=args.seed)

    os.makedirs(args.o, exist_ok=True)
    model_dir = os.path.join(args.o, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_name = f'best_denoise.pth'
    best_path = os.path.join(model_dir, model_name)

    train_dataset = TrainDataset(args.tf, args.n, args.ml, args.mi, args.mb, args.p)
    valid_dataset = ValidDataset(args.vf, args.vn, args.n, args.mml)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.b, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)
    net.apply(xavier_init)

    optimizer = optim.Adam(net.parameters(), lr=args.l)

    best_metric = -1e9
    best_epoch  = 0
    bad_count   = 0

    for epoch in range(1, args.e + 1):
        net.train()
        step = 0
        acc_loss_corrector = 0.0
        acc_loss_recommender = 0.0
        tic = time.time()

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            seq, masked_seq, random_modified_seqs, l1_ground_truth, rec_loss_mask = batch
            seq = seq.to(device, non_blocking=True)
            masked_seq = masked_seq.to(device, non_blocking=True)
            random_modified_seqs = random_modified_seqs.to(device, non_blocking=True)
            l1_ground_truth = l1_ground_truth.to(device, non_blocking=True)
            rec_loss_mask = rec_loss_mask.to(device, non_blocking=True)

            # corrector loss (noise)
            full_layer_output, padding_mask = net.corrector_forward(random_modified_seqs)
            l1_loss = net.corrector_loss(full_layer_output, l1_ground_truth, padding_mask)
            loss1 = l1_loss.sum() / (random_modified_seqs != 0).sum()
            acc_loss_corrector += float(loss1.detach().cpu().item())

            # build corrected seq (no grad)
            net.eval()
            with torch.no_grad():
                decisions_ori = net.corrector_inference(seq)
                modified_seqs = net.seq_correction(decisions_ori, seq)
                modified_masked_seqs, modified_rec_loss_mask, modified_ori_seqs = seqs_normalization(
                    modified_seqs, args.mml, args.n, args.mb
                )
            net.train()

            modified_masked_seqs = modified_masked_seqs.to(device, non_blocking=True)
            modified_rec_loss_mask = modified_rec_loss_mask.to(device, non_blocking=True)
            modified_ori_seqs = modified_ori_seqs.to(device, non_blocking=True)

            # recommender loss (orig)
            recommender_output = net.forward(masked_seq)
            rec_loss_mat = net.recommender_loss(recommender_output, rec_loss_mask, seq)
            denom1 = torch.clamp((rec_loss_mask != 0).sum(), min=1)
            rec_loss = rec_loss_mat.sum() / denom1

            # recommender loss (corrected)
            modified_recommender_output = net.forward(modified_masked_seqs.long())
            modified_loss_mat = net.recommender_loss(modified_recommender_output, modified_rec_loss_mask, modified_ori_seqs)
            denom2 = torch.clamp((modified_rec_loss_mask != 0).sum(), min=1)
            modified_rec_loss = modified_loss_mat.sum() / denom2

            total_rec_loss = rec_loss + modified_rec_loss
            acc_loss_recommender += float(total_rec_loss.detach().cpu().item())

            # contrastive loss
            cts_loss = net.contrast_loss(seq, modified_ori_seqs, random_modified_seqs)

            loss = loss1 + total_rec_loss + cts_loss * 0.001
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            if step % args.ls == 0:
                print(f'[DENOISE] epoch {epoch} step {step} '
                      f'corrector {acc_loss_corrector/step:.4f} '
                      f'rec {acc_loss_recommender/step:.4f} '
                      f'time {int(time.time()-tic)}s')

        print(f'[DENOISE] epoch {epoch} corrector {acc_loss_corrector/max(1,step):.4f} '
              f'rec {acc_loss_recommender/max(1,step):.4f}')

        if epoch % args.val_freq == 0:
            reports, key_metric = validate(net, valid_loader, args.n, args.mml, device, eval_mode=args.eval_mode)
            print(f'[DENOISE][VALID] epoch {epoch} key(ndcg@10@orig)={key_metric:.6f}')
            for m, rep in reports.items():
                print(f'  [{m}] {rep["metrics"]}')

            if key_metric > best_metric:
                best_metric = key_metric
                best_epoch = epoch
                bad_count = 0
                torch.save(net.state_dict(), best_path)
                print(f'>>> [DENOISE] New best saved @ epoch {epoch}, key={best_metric:.6f}')
            else:
                bad_count += 1
                if bad_count >= args.patience:
                    print(f'*** [DENOISE] Early stop. Best @ epoch {best_epoch}, key={best_metric:.6f}')
                    break

    return best_path, device


def train_with_debias(args, model_name='best_debias_then_rec.pth'):
    os.makedirs(args.o, exist_ok=True)
    model_dir = os.path.join(args.o, 'model')
    os.makedirs(model_dir, exist_ok=True)

    best_final_path  = os.path.join(model_dir, model_name)
    best_debias_path = os.path.join(model_dir, 'best_debiaser_only.pth')

    train_dataset = TrainDataset(args.tf, args.n, args.ml, args.mi, args.mb, args.p)
    valid_dataset = ValidDataset(args.vf, args.vn, args.n, args.mml)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.b, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # teacher (unbiased)
    best_unbiased_path = os.path.join(model_dir, 'unbiased_best.pth')
    best_unbiased_path = train_unbiased_recommender(args)

    teacher = UnbiasedModel(args.dr, args.hd, args.n, args.hn, args.ln, clip_M=args.clip_M).to(device)
    teacher.load_state_dict(torch.load(best_unbiased_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # model to train
    net = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)
    net.apply(xavier_init)

    # helper to freeze debiaser params by keywords (same as you used)
    def set_requires_grad_by_keywords(net, keywords, requires_grad: bool):
        for name, param in net.named_parameters():
            if any(k in name.lower() for k in keywords):
                param.requires_grad_(requires_grad)

    # keywords (keep yours)
    DEBIAS_KEYS = ['debias', 'correct', 'bias', 'l2']

    # Stage1: train debiaser only (loss via teacher hard labels)
    print('========== Debias Stage1: Train Debiaser Only ==========')
    optimizer = optim.Adam(net.parameters(), lr=args.l * 0.1)

    best_val = 1e18
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.e + 1):
        net.train()
        tic = time.time()
        acc = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            seq, masked_seq, random_modified_seqs, l1_ground_truth, rec_loss_mask = batch
            seq = seq.to(device, non_blocking=True)
            rec_loss_mask = rec_loss_mask.to(device, non_blocking=True)

            l2_ground_truth = teacher.debias_hard_labels(seq, rec_loss_mask, args.n - 1, bias_q = args.rho)
            debias_full_output, padding_mask = net.debiaser_forward(seq)
            l2_loss = net.corrector_loss(debias_full_output, l2_ground_truth, padding_mask)

            denom = torch.clamp((seq != 0).sum(), min=1)
            loss = l2_loss.sum() / denom

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            acc += float(loss.detach().cpu().item())
            if step % args.ls == 0:
                print(f'[Debias-S1] epoch {epoch} step {step} loss {acc/step:.4f} time {int(time.time()-tic)}s')

        # validate debiaser loss on valid set (same logic you had: create mask from masked_seq)
        if epoch % args.val_freq == 0:
            net.eval()
            losses = []
            denoms = []
            mask_token = args.n - 1
            for vb in valid_loader:
                vseq, vneg, vmasked, vleave, vtarget = vb
                vseq = vseq.to(device, non_blocking=True)
                vmasked = vmasked.to(device, non_blocking=True)
                rec_loss_mask_v = (vmasked == mask_token).float()

                l2_gt_v = teacher.debias_hard_labels(vseq, rec_loss_mask_v, mask_token, bias_q = args.rho)
                deb_out_v, pad_v = net.debiaser_forward(vseq)
                l2_v = net.corrector_loss(deb_out_v, l2_gt_v, pad_v)

                denom = max(1, int((vseq != 0).sum().item()))
                losses.append(float(l2_v.sum().item()))
                denoms.append(denom)

            val_loss = float(np.sum(losses) / np.sum(denoms))
            print(f'[Debias-S1][VALID] epoch {epoch} debias_loss={val_loss:.6f}')

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad = 0
                torch.save(net.state_dict(), best_debias_path)
                print(f'>>> [Debias-S1] New best debiaser @ epoch {epoch}, loss={best_val:.6f}')
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'*** [Debias-S1] Early stop. Best @ epoch {best_epoch}, loss={best_val:.6f}')
                    break

    net.load_state_dict(torch.load(best_debias_path, map_location=device))

    # Stage2: train recommender with debiased sequences, freeze debiaser
    print('========== Debias Stage2: Train Recommender (freeze debiaser) ==========')
    set_requires_grad_by_keywords(net, DEBIAS_KEYS, requires_grad=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.l)

    best_metric = -1e9
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.e + 1):
        net.train()
        tic = time.time()
        acc = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            seq, masked_seq, random_modified_seqs, l1_ground_truth, rec_loss_mask = batch
            seq = seq.to(device, non_blocking=True)
            masked_seq = masked_seq.to(device, non_blocking=True)
            rec_loss_mask = rec_loss_mask.to(device, non_blocking=True)

            # debiased seq (no grad)
            net.eval()
            with torch.no_grad():
                decisions_debias = net.debiaser_inference(seq)
                debiased_seqs = net.seq_correction(decisions_debias, seq)
                deb_masked, deb_rec_mask, deb_ori = seqs_normalization(debiased_seqs, args.mml, args.n, args.mb)
            net.train()

            deb_masked = deb_masked.to(device, non_blocking=True)
            deb_rec_mask = deb_rec_mask.to(device, non_blocking=True)
            deb_ori = deb_ori.to(device, non_blocking=True)

            # rec loss (orig)
            out1 = net.forward(masked_seq)
            mat1 = net.recommender_loss(out1, rec_loss_mask, seq)
            denom1 = torch.clamp((rec_loss_mask != 0).sum(), min=1)
            loss1 = mat1.sum() / denom1

            # rec loss (debiased)
            out2 = net.forward(deb_masked.long())
            mat2 = net.recommender_loss(out2, deb_rec_mask, deb_ori)
            denom2 = torch.clamp((deb_rec_mask != 0).sum(), min=1)
            loss2 = mat2.sum() / denom2

            loss = loss1 + loss2
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            acc += float(loss.detach().cpu().item())
            if step % args.ls == 0:
                print(f'[Debias-S2] epoch {epoch} step {step} loss {acc/step:.4f} time {int(time.time()-tic)}s')

        if epoch % args.val_freq == 0:
            reports, key_metric = validate(net, valid_loader, args.n, args.mml, device, eval_mode=args.eval_mode)
            print(f'[Debias-S2][VALID] epoch {epoch} key(ndcg@10@orig)={key_metric:.6f}')
            for m, rep in reports.items():
                print(f'  [{m}] {rep["metrics"]}')

            if key_metric > best_metric:
                best_metric = key_metric
                best_epoch = epoch
                bad = 0
                torch.save(net.state_dict(), best_final_path)
                print(f'>>> [Debias-S2] New best saved @ epoch {epoch}, key={best_metric:.6f}')
            else:
                bad += 1
                if bad >= args.patience:
                    print(f'*** [Debias-S2] Early stop. Best @ epoch {best_epoch}, key={best_metric:.6f}')
                    break

    return best_final_path, device


@torch.no_grad()
def test(args, best_path, device, decision_name=None):
    test_dataset = TestDataset(args.ef, args.en, args.n, args.mml)
    test_loader  = Data.DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    net = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)
    net.load_state_dict(torch.load(best_path, map_location=device))
    net.eval()

    total_result = []
    test_results = []
    for batch in test_loader:
        seq, test_neg, masked_seq, leave_one_seq, target = batch
        seq = seq.to(device, non_blocking=True)
        test_neg = test_neg.to(device, non_blocking=True)
        masked_seq = masked_seq.to(device, non_blocking=True)
        leave_one_seq = leave_one_seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.eval_mode == 'orig':
            out = net.forward(masked_seq)
            out = out[masked_seq == args.n - 1]
        else:
            decisions_ori = net.corrector_inference(leave_one_seq)
            mod_seqs = net.seq_correction(decisions_ori, leave_one_seq)
            mod_masked = seqs_normalization_test(mod_seqs, args.mml, args.n).to(device, non_blocking=True)
            out = net.forward(mod_masked)
            out = out[mod_masked == args.n - 1]

        test_results.append((seq.cpu().numpy(), out.detach().cpu().numpy(), target.detach().cpu().numpy()))

        total_result.extend(evaluate_function(out, target, test_neg))

    metrics = {
        'recall@5' : get_metrics('recall@5',  total_result),
        'recall@10': get_metrics('recall@10', total_result),
        'recall@20': get_metrics('recall@20', total_result),
        'mrr@5'    : get_metrics('mrr@5',     total_result),
        'mrr@10'   : get_metrics('mrr@10',    total_result),
        'mrr@20'   : get_metrics('mrr@20',    total_result),
        'ndcg@5': get_metrics('ndcg@5', total_result),
        'ndcg@10': get_metrics('ndcg@10', total_result),
        'ndcg@20': get_metrics('ndcg@20', total_result),
    }
    metrics['sum'] = sum(metrics.values())
    print(f'[TEST][{decision_name or "model"}] mode={args.eval_mode} metrics={metrics}')
    with open(os.path.join(args.o, f'test_results_{decision_name or "model"}.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    return metrics


@torch.no_grad()
def test_unbiased(args, best_path, device):
    test_dataset = TestDataset(args.ef, args.en, args.n, args.mml)
    test_loader  = Data.DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    net = UnbiasedModel(args.dr, args.hd, args.n, args.hn, args.ln, clip_M=args.clip_M).to(device)
    net.load_state_dict(torch.load(best_path, map_location=device))
    net.eval()

    total_result = []
    for batch in test_loader:
        seq, test_neg, masked_seq, leave_one_seq, target = batch
        test_neg = test_neg.to(device, non_blocking=True)
        masked_seq = masked_seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = net.forward(masked_seq)
        out = logits[masked_seq == args.n - 1]
        total_result.extend(evaluate_function(out, target, test_neg))

    metrics = {
        'recall@5' : get_metrics('recall@5',  total_result),
        'recall@10': get_metrics('recall@10', total_result),
        'recall@20': get_metrics('recall@20', total_result),
        'mrr@5'    : get_metrics('mrr@5',     total_result),
        'mrr@10'   : get_metrics('mrr@10',    total_result),
        'mrr@20'   : get_metrics('mrr@20',    total_result),
        'ndcg@5'   : get_metrics('ndcg@5',    total_result),
        'ndcg@10'  : get_metrics('ndcg@10',   total_result),
        'ndcg@20'  : get_metrics('ndcg@20',   total_result),
    }
    metrics['sum'] = sum(metrics.values())
    print(f'[TEST][unbiased] metrics={metrics}')
    return metrics


def mixture_logp_true(gate_logits, logits_a, logits_b, target_ids):
    logp_a = logits_a - torch.logsumexp(logits_a, dim=-1, keepdim=True)
    logp_b = logits_b - torch.logsumexp(logits_b, dim=-1, keepdim=True)

    logp_a_t = logp_a.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    logp_b_t = logp_b.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    log_g   = -F.softplus(-gate_logits)   # log(sigmoid)
    log_1mg = -F.softplus(gate_logits)    # log(1-sigmoid)

    logp_mix_t = torch.logaddexp(log_g + logp_a_t, log_1mg + logp_b_t)
    return logp_mix_t


def poe_fuse_logprob(gate_logits, logits_a, logits_b):
    g = torch.sigmoid(gate_logits).unsqueeze(-1)
    logp_a = logits_a - torch.logsumexp(logits_a, dim=-1, keepdim=True)
    logp_b = logits_b - torch.logsumexp(logits_b, dim=-1, keepdim=True)
    fused_logp = g * logp_a + (1.0 - g) * logp_b
    return fused_logp


@torch.no_grad()
def validate_merger_fusion(args, merger, expertA, expertB, valid_loader, device):
    merger.eval()
    expertA.eval()
    expertB.eval()

    MASK = args.n - 1
    total_result = []

    for batch in valid_loader:
        seq, neg, masked_seq, leave_one_seq, target = batch
        neg = neg.to(device, non_blocking=True)
        masked_seq = masked_seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits_a = expertA.forward(masked_seq)
        logits_b = expertB.forward(masked_seq)
        gate_logits = merger(masked_seq)
        fused_logp = poe_fuse_logprob(gate_logits, logits_a, logits_b)
        out = fused_logp[masked_seq == MASK]

        total_result.extend(evaluate_function(out, target, neg))

    metrics = {
        'recall@5' : get_metrics('recall@5',  total_result),
        'recall@10': get_metrics('recall@10', total_result),
        'recall@20': get_metrics('recall@20', total_result),
        'mrr@5'    : get_metrics('mrr@5',     total_result),
        'mrr@10'   : get_metrics('mrr@10',    total_result),
        'mrr@20'   : get_metrics('mrr@20',    total_result),
        'ndcg@5'   : get_metrics('ndcg@5',    total_result),
        'ndcg@10'  : get_metrics('ndcg@10',   total_result),
        'ndcg@20'  : get_metrics('ndcg@20',   total_result),
    }
    metrics['sum'] = sum(metrics.values())
    return metrics, metrics['ndcg@10']


def train_merger(args, denoise_ckpt, debias_ckpt, save_name='best_seq_merger.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join(args.o, 'model')
    os.makedirs(model_dir, exist_ok=True)

    best_path = os.path.join(model_dir, save_name)

    # -------- sanity check --------
    assert denoise_ckpt is not None and os.path.exists(denoise_ckpt), f"Missing denoise_ckpt: {denoise_ckpt}"
    assert debias_ckpt  is not None and os.path.exists(debias_ckpt),  f"Missing debias_ckpt: {debias_ckpt}"
    assert os.path.exists(args.tf), f"Missing train file: {args.tf}"
    assert os.path.exists(args.vf), f"Missing valid file: {args.vf}"
    assert os.path.exists(args.vn), f"Missing valid neg file: {args.vn}"

    # -------- loaders --------
    train_dataset = TrainDataset(args.tf, args.n, args.ml, args.mi, args.mb, args.p)
    train_loader  = Data.DataLoader(
        train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True
    )

    valid_dataset = ValidDataset(args.vf, args.vn, args.n, args.mml)
    valid_loader  = Data.DataLoader(
        valid_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True
    )

    # -------- experts (frozen) --------
    expertA = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)  # denoise expert
    expertA.load_state_dict(torch.load(denoise_ckpt, map_location=device))
    expertA.eval()
    for p in expertA.parameters():
        p.requires_grad_(False)

    expertB = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)  # debias expert
    expertB.load_state_dict(torch.load(debias_ckpt, map_location=device))
    expertB.eval()
    for p in expertB.parameters():
        p.requires_grad_(False)

    # -------- merger --------
    merger = SeqMerger(
        item_num=args.n,
        hidden=getattr(args, 'merger_hidden', 256),
        head_num=getattr(args, 'merger_hn', 2),
        layer_num=getattr(args, 'merger_ln', 2),
        max_pos=max(args.ml, args.mml),
        dropout=getattr(args, 'merger_dropout', 0.2)
    ).to(device)

    opt = optim.Adam(merger.parameters(), lr=getattr(args, 'merger_lr', 1e-5))

    epochs   = getattr(args, 'merger_epochs', 1000)
    log_step = args.ls
    val_freq = getattr(args, 'merger_val_freq', args.val_freq)
    patience = getattr(args, 'merger_patience', args.patience)

    best_metric = -1e18
    best_epoch  = 0
    bad = 0
    MASK = args.n - 1

    for epoch in range(1, epochs + 1):
        merger.train()
        tic = time.time()
        acc = 0.0
        step = 0

        for batch in train_loader:
            step += 1
            opt.zero_grad()

            # TrainDataset batch:
            #   seq, masked_seq, random_modified_seqs, l1_ground_truth, rec_loss_mask
            seq, masked_seq, _rms, _l1gt, rec_loss_mask = batch
            seq = seq.to(device, non_blocking=True)                 # (B,L)
            masked_seq = masked_seq.to(device, non_blocking=True)   # (B,L)
            rec_loss_mask = rec_loss_mask.to(device, non_blocking=True).float()  # (B,L)

            # 仅在监督位置（mask 位）训练 gate
            valid = ((seq != 0).float() * rec_loss_mask)            # (B,L)
            mask_pos = (valid > 0)
            if mask_pos.sum().item() == 0:
                continue

            with torch.no_grad():
                logits_a = expertA.forward(masked_seq)              # (B,L,V)
                logits_b = expertB.forward(masked_seq)              # (B,L,V)

            gate_logits = merger(masked_seq)                        # (B,L)

            gl = gate_logits[mask_pos]                              # (N,)
            la = logits_a[mask_pos]                                 # (N,V)
            lb = logits_b[mask_pos]                                 # (N,V)
            t  = seq[mask_pos]                                      # (N,)

            # maximize log p_mix(true)  <=> minimize -log p_mix(true)
            logp_mix_t = mixture_logp_true(gl, la, lb, t)           # (N,)
            loss = (-logp_mix_t).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(merger.parameters(), 5.0)
            opt.step()

            acc += float(loss.detach().cpu().item())
            if step % log_step == 0:
                print(f'[Merger] epoch={epoch} step={step} loss={acc/step:.4f} time={int(time.time()-tic)}s')

        print(f'[Merger] epoch={epoch} avg_loss={acc/max(1,step):.6f}')

        if epoch % val_freq == 0:
            v_metrics, v_key = validate_merger_fusion(args, merger, expertA, expertB, valid_loader, device)
            print(f'[Merger][VALID] epoch={epoch} key={v_key:.6f} metrics={v_metrics}')

            if v_key > best_metric:
                best_metric = v_key
                best_epoch = epoch
                bad = 0
                torch.save(merger.state_dict(), best_path)
                print(f'>>> [Merger] New best @ epoch={epoch}, key={best_metric:.6f} -> {best_path}')
            else:
                bad += 1
                print(f'[Merger] No improvement for {bad} validation(s).')
                if bad >= patience:
                    print(f'*** [Merger] Early stop. Best @ epoch={best_epoch}, key={best_metric:.6f}')
                    break

    if not os.path.exists(best_path):
        torch.save(merger.state_dict(), best_path)

    return best_path, device


@torch.no_grad()
def test_merger(args, merger_ckpt, denoise_ckpt, debias_ckpt, device, eval_mode='orig'):
    assert eval_mode in ['orig'], "当前 test_merger 只支持 eval_mode='orig'"

    assert merger_ckpt is not None and os.path.exists(merger_ckpt), f"Missing merger_ckpt: {merger_ckpt}"
    assert denoise_ckpt is not None and os.path.exists(denoise_ckpt), f"Missing denoise_ckpt: {denoise_ckpt}"
    assert debias_ckpt  is not None and os.path.exists(debias_ckpt),  f"Missing debias_ckpt: {debias_ckpt}"
    assert os.path.exists(args.ef), f"Missing test file: {args.ef}"
    assert os.path.exists(args.en), f"Missing test neg file: {args.en}"

    test_dataset = TestDataset(args.ef, args.en, args.n, args.mml)
    test_loader  = Data.DataLoader(
        test_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True
    )

    # experts (frozen)
    expertA = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)
    expertA.load_state_dict(torch.load(denoise_ckpt, map_location=device))
    expertA.eval()
    for p in expertA.parameters():
        p.requires_grad_(False)

    expertB = model(args.dr, args.hd, args.n, args.hn, args.ln).to(device)
    expertB.load_state_dict(torch.load(debias_ckpt, map_location=device))
    expertB.eval()
    for p in expertB.parameters():
        p.requires_grad_(False)

    merger = SeqMerger(
        item_num=args.n,
        hidden=getattr(args, 'merger_hidden', 256),
        head_num=getattr(args, 'merger_hn', 2),
        layer_num=getattr(args, 'merger_ln', 2),
        max_pos=max(args.ml, args.mml),
        dropout=getattr(args, 'merger_dropout', 0.2)
    ).to(device)
    merger.load_state_dict(torch.load(merger_ckpt, map_location=device))
    merger.eval()

    MASK = args.n - 1
    total_result = []

    for batch in test_loader:
        seq, test_neg, masked_seq, leave_one_seq, target = batch
        test_neg   = test_neg.to(device, non_blocking=True)
        masked_seq = masked_seq.to(device, non_blocking=True)
        target     = target.to(device, non_blocking=True)

        logits_a = expertA.forward(masked_seq)                    # (B,L,V)
        logits_b = expertB.forward(masked_seq)                    # (B,L,V)
        gate_logits = merger(masked_seq)                          # (B,L)

        fused_logp = poe_fuse_logprob(gate_logits, logits_a, logits_b)  # (B,L,V) logprob-like
        out = fused_logp[masked_seq == MASK]                      # (N,V)

        total_result.extend(evaluate_function(out, target, test_neg))

    metrics = {
        'recall@5' : get_metrics('recall@5',  total_result),
        'recall@10': get_metrics('recall@10', total_result),
        'recall@20': get_metrics('recall@20', total_result),
        'mrr@5'    : get_metrics('mrr@5',     total_result),
        'mrr@10'   : get_metrics('mrr@10',    total_result),
        'mrr@20'   : get_metrics('mrr@20',    total_result),
        'ndcg@5'   : get_metrics('ndcg@5',    total_result),
        'ndcg@10'  : get_metrics('ndcg@10',   total_result),
        'ndcg@20'  : get_metrics('ndcg@20',   total_result),
    }
    metrics['sum'] = sum(metrics.values())
    print(f'[TEST-MERGER] {metrics}')
    return metrics


if __name__ == '__main__':
    mode = 'test'
    parser = build_parser()
    base_args = parser.parse_args()
    args = copy.deepcopy(base_args)
    args = init_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(args.o, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # -------- train experts --------
    if mode == 'train':
        denoise_best_path, _ = train_with_denoise(args)                    
        debias_best_path,  _ = train_with_debias(args)                    

    denoise_best_path = os.path.join(model_dir, 'best_denoise.pth')
    debias_best_path  = os.path.join(model_dir, 'best_debias_then_rec.pth')

    # -------- train merger --------
    if mode == 'train':
        merger_ckpt, _ = train_merger(args, denoise_best_path, debias_best_path, save_name='best_seq_merger.pth')
    merger_ckpt = os.path.join(model_dir, 'best_seq_merger.pth')

    # # # -------- test experts --------
    print("Denoise Model Testing:")
    m1 = test(args, denoise_best_path, device, decision_name='denoise')

    print("Debias Model Testing:")
    m2 = test(args, debias_best_path, device, decision_name='debias')

    # # -------- test fusion --------
    print("Final Performance (SeqMerger Fusion):")
    m = test_merger(args, merger_ckpt, denoise_best_path, debias_best_path, device, eval_mode='orig')
