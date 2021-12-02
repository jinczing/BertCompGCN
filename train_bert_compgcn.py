import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
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
from CompGCN.utils import TrainDataset, TestDataset
from collections import defaultdict as ddict
from itertools import combinations
import random

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='hfl/chinese-macbert-base',
                    choices=['hfl/chinese-macbert-base', 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='agri_doc', choices=['agri_doc'])
parser.add_argument('--rel_dir', type=str, default='./rel')
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='compgcn', choices=['compgcn'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--train_val_ratio', type=float, default=0.9)

# compgcn arguments
parser.add_argument('--name', default='test_run', help='Set run name for saving/restoring models')
parser.add_argument('--score_func', dest='score_func', default='conve',
                    help='Score Function for Link prediction')
parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')
parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to construct batches')
parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')
parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                    help='Number of basis relation vectors to use')
parser.add_argument('--init_dim', dest='init_dim', default=100, type=int,
                    help='Initial dimension size for entities and relations')
parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
parser.add_argument('--embed_dim', dest='embed_dim', default=None, type=int,
                    help='Embedding dimension to give as input to score function')
parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.5, type=float, help='Dropout to use in GCN Layer')
parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

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
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
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
# Model


# Data Preprocess
paths = [
    'key_pims','key_sims','ne_pims','ne_sims','ws_pims','ws_sims',
'bert_similarities','bert_features','key_inclusions','ne_inclusions',
'ws_inclusions','matching_table','key_tf_idfs','ne_tf_idfs','ws_tf_idfs',
'key_berts','ne_berts','ws_berts'
]
data_paths = {}
for path in os.listdir(args.rel_path):
    data_paths[path] = os.path.join(args.rel_path, path)

g, edge_type, edge_weight, edge_norm, features, word_num, doc_num, doc_mask, test_mask = load_multi_relations_corpus(data_paths)
edge_type = torch.tensor(edge_type).to(device)
edge_weight = torch.Tensor(edge_weight).to(device)
edge_norm = torch.Tensor(edge_norm).to(device)


# compute number of real train/val/test/word nodes and number of classes
nb_node = g.num_nodes()
nb_edge = len(edge_type)
nb_train, nb_val = doc_num*args.train_val_ratio, doc_num*(1-args.train_val_ratio)
nb_test
nb_word = word_num

# instantiate model according to class number
model = CompGCN_ConvE_W(num_ent=nb_node, num_rel=nb_edge, num_base=args.num_bases,
                        init_dim=args.init_dim, gcn_dim=args.gcn_dim, embed_dim=args.embed_dim,
                        n_layer=args.gcn_layers, edge_type=edge_type, edge_norm=edge_norm,
                        bias=args.bias, gcn_drop=args.gcn_drop, opn=args.opn,
                        hid_drop=args.hid_drop, input_drop=args.input_drop,
                        conve_hid_drop=args.conve_hid_drop, feat_drop=args.feat_drop,
                        num_filt=args.num_filt, ker_sz=args.ker_sz, k_h=args.k_h, k_w=args.k_w)
model.to(device)

bert_model = BertModel()
bert_model.to(device)
if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=device)
    bert_model.load_state_dict(ckpt['bert_model'])


# load documents and compute input encodings
with open('./SDR/data/datasets/agricultures_public_only/raw_data', newline="") as f:
  reader = csv.reader(f)
  all_articles = list(reader)

docs = []
for a_id, article in enumerate(tqdm.tqdm(all_articles)):
  if not a_id:
    continue
  title, sections = article[0], ast.literal_eval(article[1])
  sections = [s[1] for s in sections]
  for i in range(len(sections)):
    sections[i] = sections[i].split('。')
  docs.append(sections)

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input

for i, doc in enumerate(docs):
    for j, sec in enumerate(doc):
        inputs = encode_input(sec, tokenizer)
        docs[i][j] = inputs

# build triplets
triplets = {
    'train_pos': [],
    'train_neg': [],
    'val_pos': [],
    'val_neg': [],
    'test': [],
}

# positive train and val
pos_ids = torch.where(edge_type==10)[0]
pos_ids = pos_ids[torch.randperm(pos_ids.size(0))]

train_pos_ids = pos_ids[:int(pos_ids.size(0)*args.train_val_ratio)]
val_pos_ids = pos_ids[int(pos_ids.size(0)*args.train_val_ratio):]

s2o_train = ddict(set)
for id in train_pos_ids:
    subj, obj = g.edges()[0][id], g.edges()[1][id]
    rel = 10
    s2o_train[(subj, rel)].add(obj)
for (subj, rel), obj in s2o_train.items():
    triplets['train_pos'].append({'triple': (subj, rel, -1), 'label': list(obj)})
s2o_val = ddict(set)
for id in val_pos_ids:
    subj, obj = g.edges()[0][id], g.edges()[1][id]
    rel = 10
    s2o_val[(subj, rel)].add(obj)
for (subj, rel), obj in s2o_val.items():
    triplets['val_pos'].append({'triple': (subj, rel, list(obj)), 'label': list(obj)})

# negative train and val
s2o_all = ddict(set)
for id in pos_ids:
    subj, obj = g.edges()[0][id], g.edges()[1][id]
    rel = 10
    s2o_all[(subj, rel)].add(obj)
all_ids = combinations(g.nodes()[doc_mask], 2)
neg_ids = []
for i, (subj, obj) in enumerate(all_ids):
    if len(s2o_all[subj, obj]) == 0:
        neg_ids.append((subj, obj))
# permute
random.shuffle(neg_ids)
train_neg_ids = neg_ids[:int(len(neg_ids)*args.train_val_ratio)]
val_neg_ids = neg_ids[int(len(neg_ids)*args.train_val_ratio):]

for subj, obj in train_neg_ids:
    triplets['train_neg'].append({'triple': (subj, rel, -1), 'label': []})
for subj, obj in val_neg_ids:
    triplets['val_neg'].append({'triple': (subj, rel, []]), 'label': []})

# test
for subj, obj in combinations(g.nodes()[tset_mask], 2):
    triplets['test'].append({'triple': (subj, rel, []), 'label': []})

data_iter = {
            'train': DataLoader(
                TrainDataset(triplets['train_pos'], triplets['train_neg'], 
                             nb_node, args),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            ),
            'val': DataLoader(
                TestDataset(triplets['val_pos'], triplets['val_neg'], 
                            nb_node, args),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            ),
            'test': DataLoader(
                TestDataset(triplets['test'], test_mask.sum(), args),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            ),
        }


# input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
# attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# document mask used for update feature
doc_mask  = train_mask + val_mask + test_mask

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
        # {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)
