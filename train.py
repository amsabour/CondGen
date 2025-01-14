import os
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict

from tqdm import tqdm

from graph_stat import *
from options import Options
from GVGAN import *
from utils import *
import pprint

from data_helper import create_graphs

from classifier.GraphSAGE import GraphSAGE
from classifier.DiffPool import DiffPool
from classifier.DGCNN import DGCNN

import math

warnings.filterwarnings("ignore")


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def load_data(graph_type):
    graphs = create_graphs(graph_type)

    train_graphs = graphs[:int(len(graphs) * 0.8)]
    test_graphs = graphs[int(len(graphs) * 0.8):]

    train_adj_mats = [nx.linalg.graphmatrix.adjacency_matrix(G).todense() for G in train_graphs]
    train_attr_vecs = []

    for G in train_graphs:
        attr_vec = np.zeros(2)
        attr_vec[G.graph['label'] - 1] = 1
        train_attr_vecs.append(attr_vec)

    test_adj_mats = [nx.linalg.graphmatrix.adjacency_matrix(G).todense() for G in test_graphs]
    test_attr_vecs = []

    for G in test_graphs:
        attr_vec = np.zeros(2)
        attr_vec[G.graph['label'] - 1] = 1
        test_attr_vecs.append(attr_vec)

    return train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs


def train(train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs, opt=None):
    training_index = list(range(0, len(train_adj_mats)))

    max_epochs = opt.max_epochs
    for epoch in range(max_epochs):
        D_real_list, D_rec_enc_list, D_rec_noise_list, D_list, Encoder_list = [], [], [], [], []
        # g_loss_list, rec_loss_list, prior_loss_list = [], [], []
        g_loss_list, rec_loss_list, prior_loss_list, aa_loss_list = [], [], [], []
        random.shuffle(training_index)
        for i in tqdm(training_index):

            ones_label = Variable(torch.ones(1)).cuda()
            zeros_label = Variable(torch.zeros(1)).cuda()
            # adj = Variable(train_adj_mats[i]).cuda()
            adj = Variable(torch.from_numpy(train_adj_mats[i]).float()).cuda()

            # if adj.shape[0] <= d_size + 2 :
            #    continue
            if adj.shape[0] <= opt.d_size + 2:
                continue
            if opt.av_size == 0:
                attr_vec = None
            else:
                # attr_vec = Variable(train_attr_vecs[i, :]).cuda()
                attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()

            # edge_num = train_adj_mats[i].sum()
            G.set_attr_vec(attr_vec)
            D.set_attr_vec(attr_vec)

            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            # print('pos_weight', pos_weight)

            mean, logvar, rec_adj = G(adj)

            noisev = torch.randn(mean.shape, requires_grad=True).cuda()
            noisev = cat_attr(noisev, attr_vec)
            rec_noise = G.decoder(noisev)

            e = int(np.sum(train_adj_mats[i])) // 2

            c_adj = topk_adj(F.sigmoid(rec_adj), e * 2)
            c_noise = topk_adj(F.sigmoid(rec_noise), e * 2)

            # train discriminator
            output = D(adj)
            errD_real = criterion_bce(output, ones_label)
            D_real_list.append(output.data.mean())
            # output = D(rec_adj)
            output = D(c_adj)
            errD_rec_enc = criterion_bce(output, zeros_label)
            D_rec_enc_list.append(output.data.mean())
            # output = D(rec_noise)
            output = D(c_noise)

            errD_rec_noise = criterion_bce(output, zeros_label)
            D_rec_noise_list.append(output.data.mean())

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            # print ("print (dis_img_loss)", dis_img_loss)
            D_list.append(dis_img_loss.data.mean())
            opt_dis.zero_grad()
            dis_img_loss.backward(retain_graph=True)
            opt_dis.step()

            # AA_loss b/w rec_adj and adj
            # aa_loss = loss_MSE(rec_adj, adj)

            loss_BCE_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_BCE_logits.cuda()

            aa_loss = loss_BCE_logits(rec_adj, adj)

            # print(c_adj,c_adj)
            # aa_loss = loss_BCE(c_adj, adj)

            # train decoder
            output = D(adj)
            errD_real = criterion_bce(output, ones_label)
            # output = D(rec_adj)
            output = D(c_adj)

            errD_rec_enc = criterion_bce(output, zeros_label)
            errG_rec_enc = criterion_bce(output, ones_label)
            # output = D(rec_noise)
            output = D(c_noise)

            errD_rec_noise = criterion_bce(output, zeros_label)
            errG_rec_noise = criterion_bce(output, ones_label)

            similarity_rec_enc = D.similarity(c_adj)
            similarity_data = D.similarity(adj)

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            # print (dis_img_loss)
            # gen_img_loss = norm*(aa_loss + errG_rec_enc  + errG_rec_noise)- dis_img_loss #- dis_img_loss #aa_loss #+ errG_rec_enc  + errG_rec_noise # - dis_img_loss
            gen_img_loss = - dis_img_loss  # norm*(aa_loss) #

            g_loss_list.append(gen_img_loss.data.mean())
            rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
            rec_loss_list.append(rec_loss.data.mean())
            # err_dec =  gamma * rec_loss + gen_img_loss

            err_dec = opt.gamma * rec_loss + gen_img_loss
            opt_dec.zero_grad()
            err_dec.backward(retain_graph=True)
            opt_dec.step()

            # train encoder
            # fix me: sum version of prior loss
            pl = []
            for j in range(mean.size()[0]):
                prior_loss = 1 + logvar[j, :] - mean[j, :].pow(2) - logvar[j, :].exp()
                prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean[j, :].data)
                pl.append(prior_loss)
            prior_loss_list.append(sum(pl))
            err_enc = sum(pl) + gen_img_loss + opt.beta * (rec_loss)  # + beta2* norm* aa_loss
            opt_enc.zero_grad()
            err_enc.backward()
            opt_enc.step()
            Encoder_list.append(err_enc.data.mean())

        print('[%d/%d]: D_real:%.4f, D_enc:%.4f, D_noise:%.4f, Loss_D:%.4f, Loss_G:%.4f, rec_loss:%.4f, prior_loss:%.4f'
              % (epoch,
                 max_epochs,
                 torch.mean(torch.stack(D_real_list)),
                 torch.mean(torch.stack(D_rec_enc_list)),
                 torch.mean(torch.stack(D_rec_noise_list)),
                 torch.mean(torch.stack(D_list)),
                 torch.mean(torch.stack(g_loss_list)),
                 torch.mean(torch.stack(rec_loss_list)),
                 torch.mean(torch.stack(prior_loss_list))))

        torch.save({'G_state_dict': G.state_dict(), 'D_state_dict': D.state_dict()}, 'my_model.pkl')

    print('Training set')
    for i in range(3):
        base_adj = train_adj_mats[i]

        if base_adj.shape[0] <= opt.d_size:
            continue
        print('Base Adj_size: ', base_adj.shape)
        attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()

        # add a new line
        G.set_attr_vec(attr_vec)

        print('Show sample')
        sample_adj = gen_adj(G, base_adj.shape[0], int(np.sum(base_adj)) // 2, attr_vec, z_size=opt.z_size)


def test(train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs):
    checkpoint = torch.load('my_model.pkl')
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])

    keys = ['LCC', 'cpl', 'gini', 'triangle_count', 'd_max']

    classifier1 = GraphSAGE(410, 2, 3, 32, 'add').cuda()
    classifier1.load_state_dict(torch.load('output/MODEL_GRIDVSTREE_GRAPHSAGE_ALL.pkl'))

    classifier2 = DiffPool(410, 2, max_num_nodes=410).cuda()
    classifier2.load_state_dict(torch.load('output/MODEL_GRIDVSTREE_DIFFPOOL_ALL.pkl'))

    classifier3 = DGCNN(410, 2, 'PROTEINS_full').cuda()
    classifier3.load_state_dict(torch.load('output/MODEL_GRIDVSTREE_DGCNN_ALL.pkl'))

    classifiers = [classifier1, classifier2, classifier3]

    acc_count_by_label = {graph_classifier: {0: 0, 1: 0} for graph_classifier in classifiers}
    original_stats_by_label = {}
    stats_by_label = {}
    counts_by_label = {}

    # Iterate through both datasets 100 times
    for epoch in range(300):
        for i in range(len(train_adj_mats)):
            base_adj = train_adj_mats[i]

            label = 0
            if train_attr_vecs[i][-1] == 1:
                label = 1

            attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()
            G.set_attr_vec(attr_vec)

            sample_adj = gen_adj(G, base_adj.shape[0], int(np.sum(base_adj)) // 2, attr_vec, z_size=opt.z_size).detach().cpu().numpy()

            original_stats = compute_graph_statistics(base_adj)
            sample_stats = compute_graph_statistics(sample_adj)

            sample_adj_tensor = torch.tensor(sample_adj).cuda()
            x = torch.eye(sample_adj_tensor.shape[0], 410).cuda()
            lower_part = torch.tril(sample_adj_tensor, diagonal=-1)
            edge_mask = (lower_part != 0).cuda()
            edges = edge_mask.nonzero().transpose(0, 1).cuda()
            edges_other_way = edges[[1, 0]]
            edges = torch.cat([edges, edges_other_way], dim=-1).cuda()
            batch = torch.zeros(sample_adj_tensor.shape[0]).long().cuda()
            graph_label = torch.tensor([label]).to('cuda').long().cuda()

            data = Bunch(x=x, edge_index=edges, batch=batch, y=graph_label, edge_weight=None)

            for graph_classifier in classifiers:
                output = graph_classifier(data)

                if output[0, label] > output[0, 1 - label]:
                    graph_classification_acc = 1
                else:
                    graph_classification_acc = 0

                acc_count_by_label[graph_classifier][label] += graph_classification_acc

            if label not in stats_by_label.keys():
                original_stats_by_label[label] = {key: 0 for key in keys}
                stats_by_label[label] = {key: 0 for key in keys}
                counts_by_label[label] = 0

            has_nan = False
            for key in keys:
                if math.isnan(sample_stats[key]):
                    has_nan = True
                    break
            if has_nan:
                continue

            for key in keys:
                original_stats_by_label[label][key] += original_stats[key]
                stats_by_label[label][key] += sample_stats[key]
            counts_by_label[label] += 1

            # if i % 150 == 1:
            #     print("(" * 30)
            #     print(label)
            #     show_graph(sample_adj, base_adj=base_adj, remove_isolated=True)
            #     print(")" * 30)

        if epoch % 2 < 3:
            for label in original_stats_by_label.keys():
                print(label, {x: original_stats_by_label[label][x] / counts_by_label[label] for x in original_stats_by_label[label].keys()})
            print('-' * 20)
            for label in stats_by_label.keys():
                print(label, {x: stats_by_label[label][x] / counts_by_label[label] for x in stats_by_label[label].keys()})
            print("=" * 20)
            for graph_classifier in acc_count_by_label.keys():
                class_0_graphs = counts_by_label[0]
                class_1_graphs = counts_by_label[1]
                print("Class 0: %.3f ----  Class 1: %.3f" % (
                    acc_count_by_label[graph_classifier][0] / class_0_graphs, acc_count_by_label[graph_classifier][1] / class_1_graphs))
            print("/" * 20)


if __name__ == '__main__':
    print('=========== OPTIONS ===========')
    pprint.pprint(vars(opt))
    print(' ======== END OPTIONS ========\n\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs = load_data('GridVSTree')

    # output_dir = opt.output_dir
    train(train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs, opt=opt)

    # test(train_adj_mats + test_adj_mats, test_adj_mats, train_attr_vecs + test_attr_vecs, test_attr_vecs)

