import sys
sys.path.append('./CompGCN')

import torch as th
import torch
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss, Precision, Recall
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from CompGCN.model import CompGCN_ConvE_W
import csv
import tqdm
import ast
from CompGCN.utils import TrainBinaryDataset, TestBinaryDataset, InferenceDataset, BinarySampler, TestBinarySampler
from collections import defaultdict as ddict
from itertools import combinations
import pickle
from torch.utils.data import DataLoader
from opencc import OpenCC
from ordered_set import OrderedSet

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=300)
parser.add_argument('--bert_init', type=str, default='hfl/chinese-macbert-base',
                    choices=['hfl/chinese-macbert-base', 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default='../ckpts/epoch=34.ckpt')
parser.add_argument('--graph_cache', type=str, default='./graph')
parser.add_argument('--triplet_cache', type=str, default='./triplet')
parser.add_argument('--init_embed_cache', type=str, default='./init_embed')
parser.add_argument('--dataset', default='agri_doc', choices=['agri_doc'])
parser.add_argument('--data_dir', type=str, default='../SDR/data/datasets/agricultures_public/raw_data')
parser.add_argument('--rel_dir', type=str, default='./rel')
parser.add_argument('--ckpt_dir', default='./ckpt', help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='compgcn', choices=['compgcn'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--train_val_ratio', type=float, default=0.9)
parser.add_argument('--resume_dir', type=str, default=None)
parser.add_argument('--test', action="store_true", default=False)

# compgcn arguments
parser.add_argument('--name', default='test_run', help='Set run name for saving/restoring models')
parser.add_argument('--score_func', dest='score_func', default='conve',
                    help='Score Function for Link prediction')
parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')
parser.add_argument('--num_workers', type=int, default=0, help='Number of processes to construct batches')
parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')
parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                    help='Number of basis relation vectors to use')
parser.add_argument('--init_dim', dest='init_dim', default=100, type=int,
                    help='Initial dimension size for entities and relations')
parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
parser.add_argument('--embed_dim', dest='embed_dim', default=200, type=int,
                    help='Embedding dimension to give as input to score function')
parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.5, type=float, help='Dropout to use in GCN Layer')
parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')

# ConvE specific hyperparameters
parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                    help='ConvE: Hidden dropout')
parser.add_argument('--feat_drop', dest='feat_drop', default=0.2, type=float, help='ConvE: Feature Dropout')
parser.add_argument('--input_drop', dest='input_drop', default=0.2, type=float, help='ConvE: Stacked Input Dropout')
parser.add_argument('--k_w', dest='k_w', default=20, type=int, help='ConvE: k_w')
parser.add_argument('--k_h', dest='k_h', default=10, type=int, help='ConvE: k_h')
parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                    help='ConvE: Number of filters in convolution')
parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
# checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

ckpt_dir = args.ckpt_dir
# if checkpoint_dir is None:
#     ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
# else:
#     ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')
device = gpu

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

if pretrained_bert_ckpt is not None:
    ckpt = torch.load(args.pretrained_bert_ckpt, map_location=device)['state_dict']
    bert_state_dict = {k.replace('model.macbert.', ''):v for k, v in ckpt.items() if 'model.macbert' in k}

import random
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, BertConfig
from transformers import AutoModel, AutoTokenizer

# Data Preprocess
data_paths = {}
for path in os.listdir(args.rel_dir):
    if os.path.isfile(os.path.join(args.rel_dir, path)):
        data_paths[os.path.basename(path).split('.')[0]] = os.path.join(args.rel_dir, path)

if not os.path.exists(args.graph_cache):
    return_tuple = load_multi_relations_doc(data_paths, logger)
    with open(args.graph_cache, 'wb') as f:
        pickle.dump(return_tuple, f)
else:
    with open(args.graph_cache, 'rb') as f:
        return_tuple = pickle.load(f)

g, edge_type, edge_weight, edge_norm, word_features, word_num, doc_num, doc_mask, test_mask, rfs, edge_type_num, gt_edge_id = return_tuple
g = g.to(device)
edge_type = torch.tensor(edge_type).to(device)
edge_weight = torch.Tensor(edge_weight).to(device)
edge_norm = edge_norm.to(device)
word_features = torch.from_numpy(word_features).to(device)
doc_mask = torch.tensor(doc_mask).bool().to(device)
test_mask = torch.tensor(test_mask).bool().to(device)
rfs = torch.from_numpy(rfs).to(device)

config = BertConfig.from_pretrained("hfl/chinese-macbert-base")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")
t2s = OpenCC('t2s').convert
bert_model = BertModel(config)
bert_model.to(device)
bert_model.load_state_dict(bert_state_dict)
args.init_dim = list(bert_model.modules())[-2].out_features # todo: setup init_dim

logger.info('loaded bert model')

# load documents and compute input encodings
with open(args.data_dir, newline="") as f:
  reader = csv.reader(f)
  all_articles = list(reader)

docs = []
sent_lens = []
titles = []
for a_id, article in enumerate(tqdm.tqdm(all_articles)):
  if not a_id:
    continue
  title, sections = article[0], ast.literal_eval(article[1])
  titles.append(title)
  sections = [s[1] for s in sections]
  sent_len = []
  for i in range(len(sections)):
    sections[i] = sections[i].split('。')
    sent_len.append([len(x)+2 for x in sections[i]])
  sent_lens.append(sent_len)
  docs.append(sections)

for i, doc in enumerate(docs):
    for j, sec in enumerate(doc):
        for k , sent in enumerate(sec):
            inputs = tokenizer(t2s(sent), max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt') # padding='max_length'
            docs[i][j][k] = inputs

logger.info('loaded docs')

logger.info('loaded data')


nb_node = g.num_nodes()
nb_edge = len(edge_type)

doc_embeds_g = None
def update_feature():
    global bert_model, model, g, doc_mask, doc_embeds_g, docs
    # bert_model.train()
    # doc_l = []
    # sents = []
    # for doc in docs:
    #     l = 0
    #     for sec in doc:
    #         for sent in sec:
    #             sents.append(sent['input_ids'].to(device))
    #         l += len(sec)
    #     doc_l.append(l)
    # sents = torch.cat(sents, dim=0)
    # bs = 512
    # outputs = []
    # with torch.no_grad():
    #     for i in tqdm.tqdm(range(sents.size(0)//bs+1)):
    #         output = list(
    #             bert_model(
    #                 sents[i*bs:(i+1)*bs],
    #                 attention_mask=None,
    #                 token_type_ids=None,
    #                 position_ids=None,
    #                 head_mask=None,
    #                 inputs_embeds=None,
    #                 output_hidden_states=False,
    #                 return_dict=False,
    #             )
    #         )[0]
    #         outputs.append(output)
    #     outputs = torch.cat(outputs, dim=0)
    #     cum = 0
    #     ds = []
    #     for i, l in enumerate(doc_l):
    #         ds.append(outputs[cum:cum+l].mean(1).mean(0).unsqueeze(0))
    #         cum += l
    #     doc_embeds_g = torch.cat(ds, dim=0).clone()
    #     print(doc_embeds_g.max(), doc_embeds_g.min())

    bert_model.eval()
    with torch.no_grad():
        doc_embeds = []
        for doc in tqdm.tqdm(docs):
            doc_embed = None
            for sec in doc:
                sec_embed = None
                for sent in sec:
                    output = list(
                        bert_model(
                            sent['input_ids'].to(device),
                            # attention_mask=sent['attention_mask'].to(device),
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=None,
                            output_hidden_states=False,
                            return_dict=False,
                        )
                    )[0].squeeze(0)
                    if sec_embed is None:
                        sec_embed = output.mean(0)
                    else:
                        sec_embed += output.mean(0)
                sec_embed /= len(sec)
                if doc_embed is None:
                    doc_embed = sec_embed
                else:
                    doc_embed += sec_embed
            doc_embed /= len(doc)
            doc_embeds.append(doc_embed.unsqueeze(0))
        doc_embeds_g = torch.cat(doc_embeds, dim=0)
if not os.path.exists(args.init_embed_cache):
    update_feature()
    with open(args.init_embed_cache, 'wb') as f:
        pickle.dump(doc_embeds_g, f)
else:
    with open(args.init_embed_cache, 'rb') as f:
        doc_embeds_g = pickle.load(f)

# instantiate model according to class number
logger.info(f'{nb_node} {nb_edge} {args.init_dim} {args.gcn_drop}')
model = CompGCN_ConvE_W(num_ent=nb_node, num_rel=edge_type_num, num_base=args.num_bases,
                        init_dim=args.init_dim, gcn_dim=args.gcn_dim, embed_dim=args.embed_dim,
                        n_layer=args.gcn_layers, edge_type=edge_type, edge_norm=edge_norm, edge_weight=edge_weight,
                        bias=args.bias, gcn_drop=args.gcn_drop, opn=args.opn,
                        hid_drop=args.hid_drop, input_drop=args.input_drop,
                        conve_hid_drop=args.conve_hid_drop, feat_drop=args.feat_drop,
                        num_filt=args.num_filt, ker_sz=args.ker_sz, k_h=args.k_h, k_w=args.k_w,
                        m=args.m, word_features=word_features, doc_features=doc_embeds_g, doc_num=doc_mask.sum().item())
model.to(device)

logger.info('loaded model')



if os.path.exists(args.triplet_cache):
    with open(args.triplet_cache, 'rb') as f:
        triplets = pickle.load(f)
else:
    # build triplets
    triplets = {
        'train_pos': [],
        'train_asym_num': None,
        'train_neg': [],
        'val_pos': [],
        'val_neg': [],
        'test': [],
    }

    # positive train and val
    pos_ids = torch.where(edge_type==gt_edge_id)[0]
    pos_ids = pos_ids[torch.randperm(pos_ids.size(0))]

    train_pos_ids = pos_ids[:int(pos_ids.size(0)*args.train_val_ratio)]
    val_pos_ids = pos_ids[int(pos_ids.size(0)*args.train_val_ratio):]
    print(g.edges()[0][pos_ids].max(), g.edges()[1][pos_ids].max(), word_num)

    s2o_all = set()
    n2e_dict = {}
    val_edge_ids = []
    for id in pos_ids:
        subj, obj = g.edges()[0][id], g.edges()[1][id]
        s2o_all.add((subj.item(), obj.item()))
        n2e_dict[(subj.item(), obj.item())] = id

    val_len = int(pos_ids.size(0)*(1-args.train_val_ratio))
    s2o_bank = s2o_all.copy()
    asym_num = 0
    while len(s2o_all):
        if len(s2o_all)>val_len:
            s = 'train_'
        else:
            s = 'val_'
        subj, obj = s2o_all.pop()
        triplets[s+'pos'].append((subj, obj, 1))
        if (obj, subj) in s2o_all:
            asym_num += 1
            s2o_all.remove((obj, subj))
            triplets[s+'pos'].append((obj, subj, 1))
            if 'val' in s:
                val_edge_ids.append(n2e_dict[(subj, obj)])
                val_edge_ids.append(n2e_dict[(obj, subj)])
        else:
            s2o_bank.add((obj, subj))
            triplets[s+'pos'].append((obj, subj, 0))
            if 'val' in s:
                val_edge_ids.append(n2e_dict[(subj, obj)])
        
    triplets['train_asym_num'] = asym_num
    triplets['val_edge_ids'] = val_edge_ids
    # for id in train_pos_ids:
    #     subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
    #     if (obj, subj) in s2o_all:
    #         s2o_all.add((subj, obj))
    #     triplets['train_pos'].append((subj.item(), obj.item(), 1))
    # for id in val_pos_ids:
    #     subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
    #     s2o_all.add((subj, obj))
    #     triplets['val_pos'].append((subj.item(), obj.item(), 1))
    
    # s2o_train = ddict(set)
    # for id in train_pos_ids:
    #     subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
    #     rel = gt_edge_id
    #     s2o_train[(subj.item(), rel)].add(obj.item())
    # for (subj, rel), obj in s2o_train.items():
    #     triplets['train_pos'].append({'triple': (subj, rel, -1), 'label': list(obj)})
    # s2o_val = ddict(set)
    # for id in val_pos_ids:
    #     subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
    #     rel = gt_edge_id
    #     s2o_val[(subj.item(), rel)].add(obj.item())
    # for (subj, rel), obj in s2o_val.items():
    #     triplets['val_pos'].append({'triple': (subj, rel, -1), 'label': list(obj)})

    logger.info('loaded pos triplets')
    print(len(s2o_bank))
    all_ids = list(combinations(g.nodes()[~test_mask].tolist(), 2)) + list(combinations(g.nodes()[~test_mask].tolist()[::-1], 2))
    print(len(all_ids))
    print(len(set(all_ids)))
    neg_ids = []
    num = 0
    for subj, obj in all_ids:
        if (subj, obj) in s2o_bank:
            num+=1
        if (subj, obj) not in s2o_bank:
            neg_ids.append((subj, obj))
    print(num)
    # random.shuffle(neg_ids)
    print(len(neg_ids))
    # neg_ids = neg_ids[list(torch.randperm(len(neg_ids)))]
    neg_set = set()
    neg_dict = {}
    for neg in neg_ids:
        s = str(neg)
        r = str(random.random())
        neg_dict[s] = r
        neg_set.add(s+'_'+r)
    val_len = int(len(neg_ids)*(1-args.train_val_ratio))
    while len(neg_set):
        # print(len(neg_ids))
        if len(neg_set)>val_len:
            s = 'train_neg'
        else:
            s = 'val_neg'
        subj, obj = ast.literal_eval(neg_set.pop().split('_')[0])
        triplets[s].append((subj, obj, 0))
        # print((obj, subj))
        neg_set.remove(str((obj, subj))+'_'+neg_dict[str((obj, subj))])
        triplets[s].append((obj, subj, 0))


    # test triplets
    




    # for i, (subj, obj) in enumerate(neg_ids):
    #     if i < :
    #         triplets['train_neg'].append((subj, obj, 0))
    #     else:
    #         triplets['val_neg'].append((subj, obj, 0))

    # negative train and val
    # s2o_all = ddict(set)
    # for id in pos_ids:
    #     subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
    #     rel = gt_edge_id
    #     s2o_all[(subj.item(), rel)].add(obj.item())
    # all_ids = combinations(g.nodes()[doc_mask]-word_num, 2)
    # neg_all = ddict(set)
    # for i, (subj, obj) in enumerate(all_ids):
    #     if len(s2o_all[(subj.item(), gt_edge_id)]) == 0 and subj.item()!=obj.item():
    #         neg_all[(subj.item(), gt_edge_id)].add(obj.item())

    # for i, ((subj, rel), obj) in enumerate(neg_all.items()):
    #     if i < int(len(neg_all.items())*args.train_val_ratio):
    #         triplets['train_neg'].append({'triple': (subj, gt_edge_id, -1), 'label': []})
    #     else:
    #         triplets['val_neg'].append({'triple': (subj, gt_edge_id, -1), 'label': []})

    # # test # todo: correct test indices
    # for subj, obj in combinations(g.nodes()[test_mask], 2):
    #     triplets['test'].append({'triple': (subj, rel, []), 'label': []})

    with open(args.triplet_cache, 'wb') as f:
        pickle.dump(triplets, f)

val_node = set()
for s, o, i in triplets['val_pos']:
    val_node.add(s)
    val_node.add(o)
# for s, o, i in triplets['val_neg']:
#     val_node.add(s)
#     val_node.add(o)

val_edge_ids = []
# for s, o, i in triplets['train_pos']:
#     if s in val_node or o in val_node:
#         val_edge_ids.append(i)
# for s, o, i in triplets['train_neg']:
#     if s in val_node or o in val_node:
#         val_edge_ids.append(len(triplets['train_pos'])+i)
# for i, (s, o) in enumerate(zip(g.edges()[0], g.edges()[1])):
#     if s.item() in val_node or o.item() in val_node:
#         ran = random.randint(0, 1)
#         if ran:
#             edge_weight[i] = 0

# edge_weight[triplets['val_edge_ids']] = 0

edge_weight[:] = 0

data_iter = {
            'train': DataLoader(
                TrainBinaryDataset(triplets['train_pos'], triplets['train_neg'], 
                             doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=BinarySampler(len(triplets['train_pos']),
                                      len(triplets['train_neg']),
                                      triplets['train_asym_num'],
                                      val_edge_ids,)
            ),
            'val': DataLoader(
                TestBinaryDataset(triplets['val_pos'], triplets['val_neg'], 
                            doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=BinarySampler(len(triplets['val_pos']),
                                      len(triplets['val_neg']),
                                      )
            ),
            # 'test': DataLoader(
            #     InferenceDataset(triplets['test'], test_mask.sum(), args),
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     num_workers=args.num_workers
            # ),
        }

test_data_iter = {
            'train': DataLoader(
                TrainBinaryDataset(triplets['train_pos'], triplets['train_neg'], 
                             doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=TestBinarySampler(len(triplets['train_pos']),
                                      len(triplets['train_neg'])),
                drop_last=True,
            ),
            'val': DataLoader(
                TestBinaryDataset(triplets['val_pos'], triplets['val_neg'], 
                            doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=TestBinarySampler(len(triplets['val_pos']),
                                      len(triplets['val_neg'])),
                drop_last=True,
            ),
        }

logger.info('loaded triplets')

# Training

# remove edges from e
edge_weight[triplets['val_edge_ids']] = 0
pos_ids = torch.where(edge_type==gt_edge_id)[0]
edge_weight[pos_ids] = 0
g.edata['weight'] = edge_weight
gt_mask = edge_weight.clone()
gt_mask[gt_mask!=0] = 1
g.edata['gt_mask'] = gt_mask



gcn_parameters = []
# for n, p in model.named_parameters():
#     if 'doc' in n: continue
#     if hasattr(p, 'parameters'):
#         print(n)
#         gcn_parameters.append(list(p.parameters()))
#     else:
#         print('pp', n)
#         gcn_parameters.append(list(p))
# gcn_parameters = sum(gcn_parameters, [])

for p in model.parameters():
    p.requires_grad = False

# model.doc_features.requires_grad = True
for p in model.parameters():
    if hasattr(p, 'requires_grad') and p.requires_grad: 
        print(p)
        continue
    gcn_parameters.append(p)
# gcn_parameters = sum(gcn_parameters, [])

for p in model.parameters():
    p.requires_grad = True


optimizer = th.optim.Adam([
        # {'params': bert_model.parameters(), 'lr': bert_lr},
        # {'params': model.doc_features, 'lr': bert_lr},
        {'params': gcn_parameters, 'lr': gcn_lr},
    ], lr=1e-3
)
if args.resume_dir is not None:
    print('resume')
    state_dict = torch.load(args.resume_dir, map_location=device)
    model_state_dict = {k:v for k, v in state_dict['model'].items() if k[-4:] != '.rel'}
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(state_dict['optimizer'])
    epoch = state_dict['epoch']

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160], gamma=0.1)




class F1_Loss(torch.nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 1
        assert y_true.ndim == 1
        device = y_pred.device
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)
        ys = []
        for pred in y_pred:
            if pred<0.5:
                ys.append(torch.Tensor([1-pred.item(), pred.item()]).to(device).unsqueeze(0))
            else:
                ys.append(torch.Tensor([1-pred.item(), pred.item()]).to(device).unsqueeze(0))
        y_pred = torch.cat(ys, dim=0)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return f1.mean()

f1_loss = F1_Loss().to(device)

def compute_doc_embeds(subjs, objs, docs, nf):
    global bert_model
    bert_model.train()

    ents = torch.cat([subjs, objs], dim=0)

    sents = []
    # attens = []
    doc_l = []
    for s in ents:
        l = 0
        for sec in docs[s]:
            for sent in sec:
                sents.append(sent['input_ids'].to(device))
                # attens.append(sent['attention_mask'].to(device))
            l += len(sec)
        doc_l.append(l)
    sents = torch.cat(sents, dim=0)
    # attens = torch.cat(attens, dim=0)
    outputs = list(
        bert_model(
            sents,
            # attention_mask=attens,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=False,
            return_dict=False,
        )
    )[0]
    cum = 0
    for i, s in enumerate(ents):
        nf[s] = outputs[cum:cum+doc_l[i]].mean(1).mean(0)
        cum += doc_l[i]

    # for s in subj:
    #     doc_embed = None
    #     for sec in docs[s]:
    #         sec_embed = None
    #         for sent in sec:
    #             output = list(
    #                     bert_model(
    #                         sent['input_ids'].to(device),
    #                         attention_mask=None,
    #                         token_type_ids=None,
    #                         position_ids=None,
    #                         head_mask=None,
    #                         inputs_embeds=None,
    #                         output_hidden_states=False,
    #                         return_dict=False,
    #                     )
    #                 )[0].squeeze(0)
    #             if sec_embed is None:
    #                 sec_embed = output.mean(0)
    #             else:
    #                 sec_embed = sec_embed + output.mean(0)
    #         sec_embed = sec_embed/len(sent)
    #         if doc_embed is None:
    #             doc_embed = sec_embed
    #         else:
    #             doc_embed = doc_embed + sec_embed
    #     doc_embed = doc_embed/len(sec)
    #     nf[s] = doc_embed
    return nf

# all_test_ids = list(combinations(g.nodes()[test_mask].tolist(), 2)) + list(combinations(g.nodes()[test_mask].tolist()[::-1], 2))
# all_test_ids = torch.tensor(all_test_ids)


iteration = 0
hards = []
hards_pos = []
def train_step(engine, batch):
    global model, bert_model, g, optimizer, docs, doc_embeds_g, rfs, device, iteration
    global hards, all_test_ids, val_node
    model.train()
    optimizer.zero_grad()


    # (triplets, labels, hards) = batch
    # triplets, labels, hards = triplets.to(device), labels.to(device), hards.to(device)
    # subj, rel = triplets[:, 0], triplets[:, 1]
    (subjs, objs, labels, ids) = batch
    subjs, objs, labels, ids = subjs.to(device), objs.to(device), labels.to(device), ids.to(device)
    # cum = 0
    syms = torch.ones_like(subjs).to(device)
    # for i in range(subjs.size(0)//2):
    #     if labels[i]+labels[i+1] == 1:
    #         syms[i] = 1
    #         syms[i+1] = 1 
    # for i, (s, o) in enumerate(zip(subjs, objs)):
    #     if s.item() in val_node or o.item() in val_node:
    #         syms[i] = 0

    rels = torch.ones_like(subjs).long().to(device)*gt_edge_id

    nf = doc_embeds_g.clone()
    # nf = compute_doc_embeds(subjs, objs, docs, doc_embeds_g.clone()).float()
    rf = rfs[subjs, objs].float() # [subj, 3]

    # preds = model(nf, g, subj, rel, rf)  # [batch_size, num_ent]
    preds = model(nf, g, subjs, rels, objs, rf)

    hard_neg_ids = torch.where((preds.flatten().round()==1) & (labels.flatten()==0))[0]
    hard_pos_ids = torch.where((preds.flatten().round()==0) & (labels.flatten()==1))[0]
    if hard_neg_ids.size(0):
        hard_ids = hard_neg_ids
        for id in hard_ids:
            hards.append(ids[id].item())
    if hard_pos_ids.size(0):
        hard_ids = hard_pos_ids
        for id in hard_ids:
            hards_pos.append(ids[id].item())

    loss = model.calc_loss(preds.flatten(), labels.flatten().float(), syms)
    
    loss.backward()
    optimizer.step()

    # with torch.no_grad():
    #     test_ids = all_test_ids[args.batch_size*iteration:args.batch_size*(iteration+1)]
    #     test_subjs, test_objs = test_ids[:,0].to(device), test_ids[:,1].to(device)
    #     test_rf = rfs[test_subjs, test_objs].float()

    #     _ = model(nf, g, test_subjs, rels, test_objs, test_rf)



    f1_score = f1_loss(preds.flatten(), labels.flatten().long()).item()
    f1_score_hard = f1_loss(preds.flatten().round(), labels.flatten().long()).item()
    if iteration%10 == 0:
        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(loss.item(), f1_score, f1_score_hard, preds.round().sum().item(), labels.sum().item()))

    iteration += 1
    

    return loss, f1_score

trainer = Engine(train_step)
if args.resume_dir:
    trainer.state.epoch = epoch


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    global triplets, doc_mask, args, hards, hards_pos

    scheduler.step()
    # update_feature()
    th.cuda.empty_cache()
    global iteration
    iteration = 0
    print('hards', len(hards))
    print('hards pos', len(hards_pos))
    data = DataLoader(
                TrainBinaryDataset(triplets['train_pos'], triplets['train_neg'], 
                             doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=BinarySampler(len(triplets['train_pos']),
                                      len(triplets['train_neg']),
                                      triplets['train_asym_num'],
                                      hards=hards.copy(),
                                      hards_pos=hards_pos.copy(),
                                      neg=len(hards)>len(hards_pos))
    )
    trainer.set_data(data)
    hards = []
    hards_pos = []

def test_step(engine, batch):
    global bert_model, model, g, doc_embeds_g, rfs, device
    with th.no_grad():
        # model.train()
        model.eval()
        # disable dropout
        # for c in model.modules():
        #     if 'drop' in c.__class__.__name__:
        #         c.eval()

        # (triplets, labels, hards) = batch
        # triplets, labels, hards = triplets.to(device), labels.to(device), hards.to(device)
        # subj, rel = triplets[:, 0], triplets[:, 1]
        (subjs, objs, labels, ids) = batch
        subjs, objs, labels, ids = subjs.to(device), objs.to(device), labels.to(device), ids.to(device)
        rels = torch.ones_like(subjs).long().to(device)*gt_edge_id

        # syms = torch.ones_like(subjs).to(device)*(-1)
        # for i in range(subjs.size(0)-1):
        #     if subjs[i] == objs[i+1] and subjs[i+1] == objs[i]:
        #         syms[i] = cum
        #         syms[i+1] = cum
        #         cum += 1

        nf = doc_embeds_g.clone().float()
        rf = rfs[subjs, objs].float()
        # preds = model(nf, g, subj, rel, rf)  # [batch_size, num_ent]
        preds = model(nf, g, subjs, rels, objs, rf)

        # preds[] = 0
        # preds[rf[:, 2]<0.5] = 0
        # preds[(rf[:, 4]<0.2)] = 0
        # preds[(rf[:, 5]<0.2)] = 0

        # if (labels==0).sum():
            # print('1', (labels==1).sum())
            # print('2', (rf[labels==1, 0]<0.5).sum())
            # print('2', (rf[labels==0, 1]>0.01).sum())
            # print('3', (rf[labels==1, 1]>0.01).sum())
            # print('4', (rf[labels==1, 2]<0.2).sum())
            # print('5', (rf[labels==1, 4]<0.1).sum())
            
            # recall
            # print('2', (rf[labels==0, 1]>0.1).sum())
            # print('3', (rf[labels==1, 1]>0.1).sum())
            # print('2', (rf[labels==0, 1]<-0.1).sum())
            # print('3', (rf[labels==1, 1]<-0.1).sum())

        # print(preds.round().sum(), hards.sum(dim=-1))
        # p = []
        # for pred in preds.flatten():
        #     if pred<0.5:
        #         p.append(torch.Tensor([pred.item(), 1-pred.item()]).unsqueeze(0).to(device))
        #     else:
        #         p.append(torch.Tensor([1-pred.item(), pred.item()]).unsqueeze(0).to(device))
        # p = torch.cat(p, dim=0)
        # print(list(preds), list(hards))
        # print((preds>0.5).sum(), hards.sum())

        return preds.flatten().round().long(), labels.flatten().long()


evaluator = Engine(test_step)
precision = Precision(average=False)
recall = Recall(average=False)
metrics={
    'precision': precision,
    'recall': recall,
    'f1': (precision * recall * 2 / (precision + recall)).mean()
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    
    train_iter = test_data_iter['train']
    evaluator.run(train_iter)
    metrics = evaluator.state.metrics
    train_precision, train_recall, train_f1 = metrics["precision"], metrics["recall"], metrics["f1"]

    val_iter = test_data_iter['val']
    evaluator.run(val_iter)
    metrics = evaluator.state.metrics
    val_precision, val_recall, val_f1 = metrics["precision"], metrics["recall"], metrics["f1"]

    logger.info(
        "Epoch: {}  Train pre: {} re: {} f1: {}  Val pre: {} re: {} f1: {}"
        .format(trainer.state.epoch, train_precision, train_recall, train_f1,
                                     val_precision, val_recall, val_f1)
    )
    if val_f1 > log_training_results.best_val_f1:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': bert_model.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                args.ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_f1 = val_f1


log_training_results.best_val_f1 = 0

train_iter = data_iter['train']
th.cuda.empty_cache()

if args.test:
    print('start testing')
    # model.train()
    batch_size = args.batch_size

    # for i in tqdm.tqdm(range(all_test_ids.size(0)//batch_size+1)):
    #     test_ids = all_test_ids[batch_size*i:batch_size*(i+1)]
    #     subjs, objs = test_ids[:,0].to(device), test_ids[:,1].to(device)
    #     # print(subjs.shape, objs.shape)
    #     # subjs, objs = subj.unsqueeze(0).to(device), obj.unsqueeze(0).to(device)
    #     rels = torch.ones_like(subjs).long().to(device)*gt_edge_id

    #     nf = doc_embeds_g.clone().float()
    #     rf = rfs[subjs, objs].float()

    #     _ = model(nf, g, subjs, rels, objs, rf)

    print(test_mask.sum())
    all_test_ids = list(combinations(g.nodes()[test_mask].tolist(), 2)) + list(combinations(g.nodes()[test_mask].tolist()[::-1], 2))
    random.shuffle(all_test_ids)
    all_test_ids = torch.tensor(all_test_ids)
    # all_test_ids_2 = torch.flip(all_test_ids, dims=(1,))
    pos_titles = []
    batch_size = args.batch_size
    model.eval()
    with th.no_grad():
        for i in tqdm.tqdm(range(all_test_ids.size(0)//batch_size+1)):
            # test_ids = torch.cat([all_test_ids[batch_size*i:batch_size*(i+1)], all_test_ids_2[batch_size*i:batch_size*(i+1)]], dim=0)
            test_ids = all_test_ids[batch_size*i:batch_size*(i+1)]
            subjs, objs = test_ids[:,0].to(device), test_ids[:,1].to(device)
            # print(subjs, objs)
            # print(subjs.shape, objs.shape)
            # subjs, objs = subj.unsqueeze(0).to(device), obj.unsqueeze(0).to(device)
            rels = torch.ones_like(subjs).long().to(device)*gt_edge_id

            nf = doc_embeds_g.clone().float()
            rf = rfs[subjs, objs].float()

            preds = model(nf, g, subjs, rels, objs, rf)
            

            preds = preds.flatten().round()
            # print(preds.sum())

            for subj, obj, pred in zip(subjs, objs, preds):
                if pred:
                    pos_titles.append([titles[subj], titles[obj]])
    with open('./inference.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Test', 'Reference'])
        writer.writerows(pos_titles)
else:
    trainer.run(train_iter, max_epochs=args.nb_epochs)
