import torch
from torch import nn
import dgl
from CompGCN.model.layer import CompGCNCov
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=5, alpha=0.2, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, sym):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target!=0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0,select.data.view(-1))
            logpt = logpt * Variable(at)

        # sym = sym*5
        # sym[sym==0] = 1

        loss = -1 * (1-pt)**self.gamma * logpt# * sym
        if self.size_average: return loss.mean()
        else: return loss.sum()


class CompGCN(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 conv_bias=True, gcn_drop=0., opn='mult'):
        super(CompGCN, self).__init__()
        self.act = torch.tanh
        # self.loss = nn.BCELoss()
        self.loss = FocalLoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.n_layer = n_layer

        self.init_embed = self.get_param([self.num_ent, self.init_dim])  # initial embedding for entities
        if self.num_base > 0:
            # linear combination of a set of basis vectors
            self.init_rel = self.get_param([self.num_base, self.init_dim])
        else:
            # independently defining an embedding for each relation
            self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])

        self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=self.num_base,
                                num_rel=self.num_rel)
        self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                opn) if n_layer == 2 else None
        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        label = label.long()
        pred = pred.unsqueeze(-1).repeat(1, 2)
        pred[0] = 1-pred[0]
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
        x = drop1(x)  # embeddings of entities [num_ent, dim]
        x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
        x = drop2(x) if self.n_layer == 2 else x
        sub_emb = torch.index_select(x, 0, subj)  # filter out embeddings of subjects in this batch
        rel_emb = torch.index_select(r, 0, rel)  # filter out embeddings of relations in this batch

        return sub_emb, rel_emb, x

class CompGCN_W(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm, 
                 edge_weight, conv_bias=True, gcn_drop=0., opn='mult', word_features=None, doc_features=None, m=None, doc_num=None):
        super(CompGCN_W, self).__init__()
        self.act = torch.tanh
        self.loss = FocalLoss()
        # self.loss = nn.BCELoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.edge_weight = edge_weight # [E]
        self.n_layer = n_layer
        self.doc_num = doc_num

        self.init_embed = self.get_param([self.num_ent-self.doc_num, self.init_dim])  # initial embedding for entities
        if self.num_base > 0:
            # linear combination of a set of basis vectors
            self.init_rel = self.get_param([self.num_base, self.init_dim])
        else:
            # independently defining an embedding for each relation
            self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])

        self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=self.num_base,
                                num_rel=self.num_rel)
        self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                opn) if n_layer == 2 else None
        self.bias = nn.Parameter(torch.zeros(self.doc_num))

        self.m = m
        # self.word_features = nn.Parameter(word_features)
        self.doc_features = nn.Parameter(doc_features)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label, sym):
        label = label.long()
        pred = pred.unsqueeze(-1).repeat(1, 2)
        pred[:, 0] = 1-pred[:, 0]
        return self.loss(pred, label, sym)

    def forward_base(self, nf, g, subj, rel, drop1=None, drop2=None):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        # x, r = nf, rf  # embedding of relations
        # print(self.word_features.shape, nf.shape)
        # torch.cat([self.word_features, nf], dim=0) torch.cat([self.word_features, self.doc_features])
        device = nf.device # torch.zeros(self.num_ent-self.doc_num, nf.size(-1)).to(device)
        x, r = nf, self.init_rel  # embedding of relations
        x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm, self.edge_weight)
        x = drop1(x)  # embeddings of entities [num_ent, dim]
        x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm, self.edge_weight) if self.n_layer == 2 else (x, r)
        x = drop2(x) if self.n_layer == 2 else x
        sub_emb = torch.index_select(x, 0, subj)  # filter out embeddings of subjects in this batch
        rel_emb = torch.index_select(r, 0, rel)  # filter out embeddings of relations in this batch

        return sub_emb, rel_emb, x


class CompGCN_DistMult(CompGCN):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0.):
        super(CompGCN_DistMult, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                               edge_type, edge_norm, bias, gcn_drop, opn)
        self.drop = nn.Dropout(hid_drop)

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score


class CompGCN_ConvE(CompGCN):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(CompGCN_ConvE, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                            edge_type, edge_norm, bias, gcn_drop, opn)
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)  # fully connected projection

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        assert self.embed_dim == self.k_h * self.k_w
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, subj, rel, self.drop, self.input_drop)
        stack_input = self.concat(sub_emb, rel_emb)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

class CompGCN_ConvE_W(CompGCN_W):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm, edge_weight,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, word_features=None, doc_features=None, m=None, doc_num=None):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(CompGCN_ConvE_W, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                            edge_type, edge_norm, edge_weight, bias, gcn_drop, opn, word_features=word_features, doc_features=doc_features, m=0.7, doc_num=doc_num)
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(self.conve_hid_drop)  # hidden layer dropout
        self.mlp_drop = torch.nn.Dropout(0.2)
        self.fc_drop = torch.nn.Dropout(0.2)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)  # fully connected projection

        self.m = m

        class BertPredictorDiff(nn.Module):
            def __init__(self, embed_dim, rf_dim=4):
                super().__init__()
                self.embed_dim = embed_dim
                self.rf_dim = rf_dim
                self.fc = torch.nn.Linear(embed_dim+rf_dim, 1)
            
            def forward(self, nf, rf, subj):
                all_ent = nf
                nb_subj = subj.size(0)
                nb_ent = all_ent.size(0)
                nf = torch.index_select(nf, 0, subj)
                x = nf.unsqueeze(1)-all_ent.unsqueeze(0)
                x = torch.cat([x, rf], dim=-1)
                x = x.view(-1, self.embed_dim+self.rf_dim)
                x = self.fc(x)
                x = torch.sigmoid(x)
                x = x.view(nb_subj, nb_ent)
                return x

        class BertPredictor(nn.Module):
            def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None):
                super().__init__()
                self.embed_dim = embed_dim
                self.k_h = k_h
                self.k_w = k_w

                self.bn0 = torch.nn.BatchNorm2d(1)
                self.bn1 = torch.nn.BatchNorm2d(num_filt)
                self.bn2 = torch.nn.BatchNorm1d(embed_dim)
                self.bn3 = torch.nn.BatchNorm1d(embed_dim)

                self.drop = torch.nn.Dropout(hid_drop)
                self.input_drop = torch.nn.Dropout(input_drop)
                self.feature_drop = torch.nn.Dropout(feat_drop)
                self.hidden_drop = torch.nn.Dropout(conve_hid_drop)

                self.flat_sz = flat_sz_h * flat_sz_w * num_filt

                self.mlp_bert = torch.nn.Linear(rel_fs, embed_dim)
                self.conv2d_bert = torch.nn.Conv2d(in_channels=1, out_channels=num_filt,
                                      kernel_size=(ker_sz, ker_sz), stride=1, padding=0, bias=bias)
                self.fc_bert = torch.nn.Linear(flat_sz, embed_dim)

                self.bias_bert = nn.Parameter(torch.zeros(num_ent))

            def concat(self, ent_embed, rel_embed):
                ent_embed = ent_embed.view(-1, 1, self.embed_dim)
                rel_embed = rel_embed.view(-1, 1, self.embed_dim)
                stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
                assert self.embed_dim == self.k_h * self.k_w
                stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
                return stack_input
            
            def forward(self, nf, rf, subj, all_ent):
                x = self.mlp_bert(rf)
                x = self.bn3(x)
                x = F.relu(x)
                # x = self.concat(torch.index_select(nf, 0, subj), x)
                x = self.concat(all_ent, x)
                x = self.bn0(x)
                x = self.conv2d_bert(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.feature_drop(x)
                x = x.view(-1, self.flat_sz)
                x = self.fc_bert(x)
                x = self.hidden_drop(x)
                x = self.bn2(x)
                x = F.relu(x)
                nf = torch.index_select(nf, 0, subj)
                x = torch.mm(nf, x.transpose(1, 0))
                x += self.bias_bert.expand_as(x)
                bert_score = torch.sigmoid(x)
                return bert_score

        # self.bert_predictor = BertPredictor(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
        #          bias, gcn_drop, opn, hid_drop, input_drop, conve_hid_drop, feat_drop,
        #          num_filt, ker_sz, k_h, k_w)

        # self.bert_predictor = BertPredictorDiff(self.init_dim)

        self.rel_mlp = torch.nn.Linear(2*self.embed_dim, self.embed_dim)
        self.rel_bn = torch.nn.BatchNorm1d(self.embed_dim)
        # self.rel_bn = torch.nn.LayerNorm(self.embed_dim)
        # self.diff_bn = torch.nn.BatchNorm1d(self.embed_dim)
        self.diff = torch.nn.Linear(self.embed_dim, 1)
        self.bert_diff = torch.nn.Linear(self.init_dim, 1)
        self.relu = torch.nn.ReLU()
        self.predictor = torch.nn.Linear((6+self.embed_dim+self.init_dim), 1)
        # self.predictor3 = torch.nn.Linear(2, 1)


    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        assert self.embed_dim == self.k_h * self.k_w
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def forward(self, nf, g, subj, rel, obj, rf):
        """
        # :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        # self.rel_bn.train()
        ent = subj.size(0)
        sub_emb, rel_emb, all_ent = self.forward_base(nf, g, subj, rel, self.drop, self.input_drop)
        
        obj_emb = torch.index_select(all_ent, index=obj, dim=0)

        # sub_emb_rel = sub_emb * rel_emb
        # sub_emb = self.relu(torch.cat([sub_emb, rel_emb], dim=-1))
        # sub_emb = self.rel_mlp(sub_emb)
        # sub_emb = self.mlp_drop(sub_emb)
        # sub_emb = self.rel_bn(sub_emb)
        # sub_emb = sub_emb*rel_emb

        # sub_emb = self.relu(sub_emb)
        # sub_emb = self.rel_mlp(torch.cat([sub_emb, rel_emb], dim=-1))
        # sub_emb = self.rel_bn(sub_emb)
        # obj_emb = self.relu(obj_emb)
        # obj_emb = self.rel_mlp(torch.cat([obj_emb, rel_emb], dim=-1))
        # obj_emb = self.rel_bn(obj_emb)

        bert_sub_emb = torch.index_select(nf, index=subj, dim=0)
        bert_obj_emb = torch.index_select(nf, index=obj, dim=0)

        # x = torch.cat([sub_emb, obj_emb], dim=-1)
        # x = self.predictor(x)
        # x = self.relu(x)
        # sub_emb = sub_emb/torch.norm(sub_emb, dim=-1, keepdim=True)
        # obj_emb = obj_emb/torch.norm(obj_emb, dim=-1, keepdim=True)
        # x = torch.bmm(sub_emb.view(ent, 1, -1), obj_emb.view(ent, -1, 1)).squeeze(-1)
        # print(ent, x.shape, rf.shape)
        # gcn_score = x.clamp(0) #torch.sigmoid(x)
        sub_emb = sub_emb/torch.norm(sub_emb, dim=-1, keepdim=True)
        obj_emb = obj_emb/torch.norm(obj_emb, dim=-1, keepdim=True)
        x = sub_emb-obj_emb
        x = x/torch.norm(x, dim=-1, keepdim=True)
        diff = self.mlp_drop(x)
        # diff = self.diff(x)
        # diff = self.relu(diff)
        # diff = diff/torch.norm(diff, dim=-1, keepdim=True)
        # diff = diff.view(subj.size(0)//2, 2, self.embed_dim)
        # diff1, diff2 = torch.chunk(diff, 2, dim=1)
        # diff = torch.bmm(diff1, diff2.permute(0, 2, 1)).squeeze(-1)
        # diff = self.relu(diff)
        # diff = self.mlp_drop(diff)

        bert_sub_emb = bert_sub_emb/torch.norm(bert_sub_emb, dim=-1, keepdim=True)
        bert_obj_emb = bert_obj_emb/torch.norm(bert_obj_emb, dim=-1, keepdim=True)
        x = bert_sub_emb-bert_obj_emb
        x = x/torch.norm(x, dim=-1, keepdim=True)
        bert_diff = self.mlp_drop(x)
        # bert_diff = self.bert_diff(x)
        # bert_diff = self.relu(bert_diff)
        # bert_diff = bert_diff/torch.norm(bert_diff, dim=-1, keepdim=True)
        # bert_diff = bert_diff.view(subj.size(0)//2, 2, self.init_dim)
        # diff1, diff2 = torch.chunk(bert_diff, 2, dim=1)
        # bert_diff = torch.bmm(diff1, diff2.permute(0, 2, 1)).squeeze(-1)
        # bert_diff = self.relu(bert_diff)
        # bert_diff = self.mlp_drop(bert_diff)

        # x = self.relu(x)
        # zero = torch.zeros_like(diff).to(diff)
        # x = rf
        # x = x.view(-1, 2*6)
        x = torch.cat([rf, diff, bert_diff], dim=-1) # 2*ent x 
        # x = self.fc_drop(x)
        # x = x.view(-1, 2*(6+self.embed_dim+self.init_dim)) # ent x 12
        # x = self.mlp_drop(x)
        x = self.predictor(x)
        # x = x.view(-1, 1)
        # x = self.predictor2(rf)
        score = torch.sigmoid(x)
        


        # gcn_score = torch.sigmoid(x)
        # x = self.predictor2(rf)
        # bert_score = torch.sigmoid(x)
        # score = self.predictor3(torch.cat([gcn_score, bert_score], dim=-1))
        # score = torch.sigmoid(score)



        # score = torch.sigmoid(x)
        
        # score =  gcn_score*self.m + bert_score*(1-self.m)
        
        # score = torch.sigmoid(x)
        # thres_score = torch.zeros_like(score).to(score.device)
        # thres_score[rf[:, 0]>0.8, 0] = score[rf[:, 0]>0.8, 0]
        # score = torch.masked_fill((rf[:, 0]<0.5).unsqueeze(-1), 0)
        # score[rf[:, 0]<0.5, 0] = 0

        return score


        
        stack_input = self.concat(sub_emb, rel_emb)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = stack_input
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        gcn_score = torch.sigmoid(x)

        # bert_score = self.bert_predictor(nf, rf, subj)

        score = gcn_score#*self.m + bert_score*(1-self.m)
        return score


if __name__ == '__main__':
    src, tgt = [0, 1, 0, 3, 2], [1, 3, 3, 4, 4]
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges(src, tgt)  # src -> tgt
    g.add_edges(tgt, src)  # tgt -> src
    edge_type = torch.tensor([0, 0, 0, 1, 1] + [2, 2, 2, 3, 3])
    import numpy as np

    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = in_deg ** -0.5
    norm[np.isinf(norm)] = 0
    g.ndata['xxx'] = norm
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    edge_norm = g.edata.pop('xxx').squeeze()
    print(edge_norm.dtype)

    distmult = CompGCN_DistMult(num_ent=5, num_rel=2, num_base=2, init_dim=10, gcn_dim=5, embed_dim=3, n_layer=2,
                                edge_type=edge_type, edge_norm=edge_norm)
    drop = nn.Dropout(0.1)
    score = distmult(g, torch.tensor([0, 4]), torch.tensor([0, 1]))
    print(score)
