import sys
sys.path.append('./CompGCN')

import torch as th
import torch
from transformers import AutoModel, AutoTokenizer
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
from CompGCN.utils import TrainBinaryDataset, TestBinaryDataset, InferenceDataset, BinarySampler
from collections import defaultdict as ddict
from itertools import combinations
import random
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, BertConfig
import pickle
from torch.utils.data import DataLoader
from opencc import OpenCC

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='hfl/chinese-macbert-base',
                    choices=['hfl/chinese-macbert-base', 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--graph_cache', type=str, default='./graph')
parser.add_argument('--triplet_cache', type=str, default='./triplet')
parser.add_argument('--init_embed_cache', type=str, default='./init_embed')
parser.add_argument('--dataset', default='agri_doc', choices=['agri_doc'])
parser.add_argument('--data_dir', type=str, default='/content/SDR/data/datasets/agricultures_public/raw_data')
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
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
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


# Data Preprocess
data_paths = {}
for path in os.listdir(args.rel_dir):
    if os.path.isfile(os.path.join(args.rel_dir, path)):
        data_paths[os.path.basename(path).split('.')[0]] = os.path.join(args.rel_dir, path)

if not os.path.exists(args.graph_cache):
    return_tuple = load_multi_relations_corpus(data_paths, logger)
    with open(args.graph_cache, 'wb') as f:
        pickle.dump(return_tuple, f)
else:
    with open(args.graph_cache, 'rb') as f:
        return_tuple = pickle.load(f)

g, edge_type, edge_weight, edge_norm, word_features, word_num, doc_num, doc_mask, test_mask, rfs = return_tuple
g = g.to(device)
edge_type = torch.tensor(edge_type).to(device)
edge_weight = torch.Tensor(edge_weight).to(device)
edge_norm = edge_norm.to(device)
word_features = torch.from_numpy(word_features).to(device)
doc_mask = torch.tensor(doc_mask).bool().to(device)
test_mask = torch.tensor(test_mask).bool().to(device)
rfs = torch.from_numpy(rfs).to(device)

logger.info('loaded data')


nb_node = g.num_nodes()
nb_edge = len(edge_type)

config = BertConfig.from_pretrained("hfl/chinese-macbert-base")
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")
t2s = OpenCC('t2s').convert
bert_model = BertModel(config)
bert_model.to(device)
if pretrained_bert_ckpt is not None:
    ckpt = torch.load(args.pretrained_bert_ckpt, map_location=device)['state_dict']
    bert_state_dict = {k.replace('model.macbert.', ''):v for k, v in s.items() if 'model.macbert' in k}
    bert_model.load_state_dict(bert_state_dict)
args.init_dim = list(bert_model.modules())[-2].out_features # todo: setup init_dim

logger.info('loaded bert model')

# instantiate model according to class number
logger.info(f'{nb_node} {nb_edge} {args.init_dim} {args.gcn_drop}')
model = CompGCN_ConvE_W(num_ent=nb_node, num_rel=nb_edge, num_base=args.num_bases,
                        init_dim=args.init_dim, gcn_dim=args.gcn_dim, embed_dim=args.embed_dim,
                        n_layer=args.gcn_layers, edge_type=edge_type, edge_norm=edge_norm, edge_weight=edge_weight,
                        bias=args.bias, gcn_drop=args.gcn_drop, opn=args.opn,
                        hid_drop=args.hid_drop, input_drop=args.input_drop,
                        conve_hid_drop=args.conve_hid_drop, feat_drop=args.feat_drop,
                        num_filt=args.num_filt, ker_sz=args.ker_sz, k_h=args.k_h, k_w=args.k_w,
                        m=args.m, word_features=word_features, doc_num=doc_mask.sum().item())
model.to(device)

logger.info('loaded model')


# load documents and compute input encodings
with open(args.data_dir, newline="") as f:
  reader = csv.reader(f)
  all_articles = list(reader)

docs = []
sent_lens = []
for a_id, article in enumerate(tqdm.tqdm(all_articles)):
  if not a_id:
    continue
  title, sections = article[0], ast.literal_eval(article[1])
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
            inputs = tokenizer(t2s(sent), max_length=args.max_length, truncation=True, padding='max_length', return_tensors='pt')
            docs[i][j][k] = inputs

logger.info('loaded docs')


if os.path.exists(args.triplet_cache):
    with open(args.triplet_cache, 'rb') as f:
        triplets = pickle.load(f)
else:
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
    # pos_ids = pos_ids[torch.randperm(pos_ids.size(0))]

    train_pos_ids = pos_ids[:int(pos_ids.size(0)*args.train_val_ratio)]
    val_pos_ids = pos_ids[int(pos_ids.size(0)*args.train_val_ratio):]
    print(g.edges()[0][pos_ids].max(), g.edges()[1][pos_ids].max(), word_num)

    s2o_train = ddict(set)
    for id in train_pos_ids:
        subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
        rel = 10
        s2o_train[(subj.item(), rel)].add(obj.item())
    for (subj, rel), obj in s2o_train.items():
        triplets['train_pos'].append({'triple': (subj, rel, -1), 'label': list(obj)})
    s2o_val = ddict(set)
    for id in val_pos_ids:
        subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
        rel = 10
        s2o_val[(subj.item(), rel)].add(obj.item())
    for (subj, rel), obj in s2o_val.items():
        triplets['val_pos'].append({'triple': (subj, rel, -1), 'label': list(obj)})

    logger.info('loaded pos triplets')

    # negative train and val
    s2o_all = ddict(set)
    for id in pos_ids:
        subj, obj = g.edges()[0][id]-word_num, g.edges()[1][id]-word_num
        rel = 10
        s2o_all[(subj.item(), rel)].add(obj.item())
    all_ids = combinations(g.nodes()[doc_mask]-word_num, 2)
    neg_all = ddict(set)
    for i, (subj, obj) in enumerate(all_ids):
        if len(s2o_all[(subj.item(), 10)]) == 0 and subj.item()!=obj.item():
            # neg_ids.append((subj.item(), obj.item()))
            neg_all[(subj.item(), 10)].add(obj.item())
    # permute
    # random.shuffle(neg_ids)
    # train_neg_ids = neg_ids[:int(len(neg_ids)*args.train_val_ratio)]
    # val_neg_ids = neg_ids[int(len(neg_ids)*args.train_val_ratio):]

    for i, ((subj, rel), obj) in enumerate(neg_all.items()):
        if i < int(len(neg_all.items())*args.train_val_ratio):
            triplets['train_neg'].append({'triple': (subj, 10, -1), 'label': []})
        else:
            triplets['val_neg'].append({'triple': (subj, 10, -1), 'label': []})
    # for subj, obj in train_neg_ids:
    #     triplets['train_neg'].append({'triple': (subj, 10, -1), 'label': []})
    # for subj, obj in val_neg_ids:
    #     triplets['val_neg'].append({'triple': (subj, 10, []), 'label': []})

    # test # todo: correct test indices
    for subj, obj in combinations(g.nodes()[test_mask], 2):
        triplets['test'].append({'triple': (subj, rel, []), 'label': []})

    with open(args.triplet_cache, 'wb') as f:
        pickle.dump(triplets, f)



data_iter = {
            'train': DataLoader(
                TrainBinaryDataset(triplets['train_pos'], triplets['train_neg'], 
                             doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=BinarySampler(len(triplets['train_pos']),
                                      len(triplets['train_neg']))
            ),
            'val': DataLoader(
                TestBinaryDataset(triplets['val_pos'], triplets['val_neg'], 
                            doc_mask.sum().item(), args),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=BinarySampler(len(triplets['val_pos']),
                                      len(triplets['val_neg']))
            ),
            # 'test': DataLoader(
            #     InferenceDataset(triplets['test'], test_mask.sum(), args),
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     num_workers=args.num_workers
            # ),
        }

logger.info('loaded triplets')

# Training
doc_embeds_g = None
def update_feature():
    global model, g, doc_mask, doc_embeds_g, docs
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
                            attention_mask=None,
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

gcn_parameters = []
for child in model.children():
    if 'BertPredictor' in child.__class__.__name__:
        continue
    gcn_parameters.append(list(child.parameters()))
gcn_parameters = sum(gcn_parameters, [])

optimizer = th.optim.Adam([
        {'params': bert_model.parameters(), 'lr': bert_lr},
        {'params': model.bert_predictor.parameters(), 'lr': bert_lr},
        {'params': gcn_parameters, 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

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
                ys.append(torch.Tensor([pred.item(), 1-pred.item()]).to(device).unsqueeze(0))
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
        return 1 - f1.mean()

f1_loss = F1_Loss().to(device)

def compute_doc_embeds(subj, docs, nf):
    sents = []
    doc_l = []
    for s in subj:
        l = 0
        for sec in docs[s]:
            for sent in sec:
                sents.append(sent['input_ids'].to(device))
            l += len(sec)
        doc_l.append(l)
    sents = torch.cat(sents, dim=0)
    outputs = list(
        bert_model(
            sents,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=False,
            return_dict=False,
        )
    )[0]
    cum = 0
    for i, s in enumerate(subj):
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
iteration = 0
def train_step(engine, batch):
    global model, bert_model, g, optimizer, docs, doc_embeds_g, rfs, device, iteration
    
    model.train()
    optimizer.zero_grad()
    (triplets, labels, hards) = batch
    triplets, labels, hards = triplets.to(device), labels.to(device), hards.to(device)
    subj, rel = triplets[:, 0], triplets[:, 1]

    # todo: inference embeddings of subj
    # nf = doc_embeds_g.clone()
    nf = compute_doc_embeds(subj, docs, doc_embeds_g.clone()).float()

    # todo: extract rf
    rf = rfs[subj, ...].float() # [subj, num_ent, 4]

    preds = model(nf, g, subj, rel, rf)  # [batch_size, num_ent]
    loss = model.calc_loss(preds, labels)
    
    loss.backward()
    optimizer.step()

    f1_score = f1_loss(preds.flatten(), hards.flatten().long())
    if iteration%50 == 0:
        print(loss, f1_score)

    iteration += 1
    

    return loss, f1_score

trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()
    global iteration
    iteration = 0

def test_step(engine, batch):
    global bert_model, g, doc_embeds_g, rfs, device
    with th.no_grad():
        model.eval()

        (triplets, labels, hards) = batch
        triplets, labels, hards = triplets.to(device), labels.to(device), hards.to(device)
        subj, rel = triplets[:, 0], triplets[:, 1]
        nf = doc_embeds_g.clone().float()
        rf = rfs[subj, ...].float()
        preds = model(nf, g, subj, rel, rf)  # [batch_size, num_ent]
        # p = []
        # for pred in preds.flatten():
        #     if pred<0.5:
        #         p.append(torch.Tensor([pred.item(), 1-pred.item()]).unsqueeze(0).to(device))
        #     else:
        #         p.append(torch.Tensor([1-pred.item(), pred.item()]).unsqueeze(0).to(device))
        # p = torch.cat(p, dim=0)
        # print(list(preds), list(hards))
        # print((preds>0.5).sum(), hards.sum())

        return preds.flatten().round().long(), hards.flatten().long()


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
    
    train_iter = data_iter['train']
    evaluator.run(train_iter)
    metrics = evaluator.state.metrics
    train_precision, train_recall, train_f1 = metrics["precision"], metrics["recall"], metrics["f1"]

    val_iter = data_iter['val']
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
if not os.path.exists(args.init_embed_cache):
    update_feature()
    with open(args.init_embed_cache, 'wb') as f:
        pickle.dump(doc_embeds_g, f)
else:
    with open(args.init_embed_cache, 'rb') as f:
        doc_embeds_g = pickle.load(f)

train_iter = data_iter['train']
trainer.run(train_iter, max_epochs=args.nb_epochs)
