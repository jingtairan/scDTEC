#!/usr/bin/env python

from .layer import *
from .model import *
from .loss import *
from .dataset import load_dataset
from .utils import *


import time
import torch

import numpy as np
import pandas as pd
import os
import scanpy as sc

import copy

import torch.nn.functional as F

from .model import  GCN, GCL
from .graph_learners import *
from sklearn.cluster import KMeans

import dgl

import random

import wandb
from tqdm import tqdm



def scDTEC_function(
        args,
        data_list,
        n_centroids = 30,
        outdir = None,
        verbose = False,
        pretrain_model = None,
        pretrain_graphlearner = None,
        lr = 0.0002,
        batch_size = 64,
        gpu = 0,
        seed = 18,
        encode_dim = [1024, 128],
        decode_dim = [],
        latent = 10,
        min_peaks = 100,
        min_cells = 3,
        n_feature = 100000,
        log_transform = False,
        max_iter = 30000,
        weight_decay = 5e-4,
        impute = False,
        binary = False,
        embed = 'UMAP',
        reference = 'cell_type',
        cluster_method = 'leiden',
        alpha = 1.0,
        beta = 1.0
    ):

    dataset = data_list

    wandb.init(
        project=dataset,
        config={
            "dataset": dataset,
            "latent": latent,
            "n_cluster": n_centroids,
            "lr": lr,
            "maskfeat_rate_learner": args.maskfeat_rate_learner,
            "maskfeat_rate_anchor": args.maskfeat_rate_anchor,
            "dropedge_rate": args.dropedge_rate,
            "knn_k": args.knn_k,
            "type_learner": args.type_learner,
            "aplha": alpha,
            "beta": beta
        },
        entity="tairan",
        name = f'scDGEC_github test'
    )

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)


    def loss_gcl(x, model, graph_learner, features, anchor_adj, alpha, beta):
        recon_loss, kl_loss = model.loss_function(x, anchor_adj)
        loss_VAE = (recon_loss + (alpha * kl_loss))/len(x)

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        _, z1, _, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        _, z2, _, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        loss = beta*loss + loss_VAE
        return loss, learned_adj, recon_loss, kl_loss

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    print("\n************************************************************************************************************")
    print("scDTEC: Unsupervised Deep Topology Embedded Characterization of Single-Cell Chromatin Accessibility Profiles")
    print("**************************************************************************************************************\n")
    

    adata, trainloader, testloader = load_dataset(
        data_list,
        min_genes=min_peaks,
        min_cells=min_cells,
        n_top_genes=n_feature,
        batch_size=batch_size, 
        log=None,
    )

    cell_num = adata.shape[0] 
    input_dim = adata.shape[1] 	
    
    k = n_centroids
    print('目前预设簇数为{}'.format(k))

    # SUBLIME data
    nclasses = k
    labels = torch.Tensor(adata.obs[reference].cat.codes)

    if outdir:
        outdir =  outdir+'/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('outdir: {}'.format(outdir))
        
    print("\n======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}'.format(
        cell_num, input_dim, max_iter, batch_size,  min_peaks,  min_cells))
    print("============================")

    dims = [input_dim,  latent,  encode_dim,  decode_dim]

    features = torch.tensor(adata.X.todense(), dtype=torch.float32)
    nfeats = features.shape[1]
    if args.downstream_task == 'clustering':
        n_clu_trials = copy.deepcopy(args.ntrials)
        args.ntrials = 1
    else:
        print("downstream wrong!")

    for trial in range(args.ntrials):

        setup_seed(trial)

        if args.gsl_mode == 'structure_inference':
            if args.sparse:
                anchor_adj_raw = torch_sparse_eye(features.shape[0])
            else:
                anchor_adj_raw = torch.eye(features.shape[0])
        elif args.gsl_mode == 'structure_refinement':
            if args.sparse:
                anchor_adj_raw = adj_original
            else:
                anchor_adj_raw = torch.from_numpy(adj_original)

        anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

        if args.sparse:
            anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
            anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

        if args.type_learner == 'fgp':
            graph_learner = FGP_learner(features.cpu(), args.knn_k, args.sim_function, 6, args.sparse)
        elif args.type_learner == 'mlp':
            graph_learner = MLP_learner(2, features.shape[1], args.knn_k, args.sim_function, 6, args.sparse,
                                        args.activation_learner)
        elif args.type_learner == 'att':
            graph_learner = ATT_learner(2, features.shape[1], args.knn_k, args.sim_function, 6, args.sparse,
                                        args.activation_learner)
        elif args.type_learner == 'gnn':
            graph_learner = GNN_learner(2, features.shape[1], args.knn_k, args.sim_function, 6, args.sparse,
                                        args.activation_learner, anchor_adj)

        model = scDTEC(dims=dims, nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                     emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                     dropout_graph=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

        optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

        if torch.cuda.is_available():
            model = model.cuda()
            graph_learner = graph_learner.cuda()
            features = features.cuda()
            labels = labels.cuda()
            if not args.sparse:
                anchor_adj = anchor_adj.cuda()

        print(model)

        if not pretrain_model:
            print('\n## Training Model ##')
            model.train()
            graph_learner.train()

            iteration = 0
            n_epoch = int(np.ceil(max_iter/len(trainloader)))

            epoch_i = 0

            with tqdm(range(max_iter), total=max_iter, desc='Epochs') as tq:
                for epoch in tq:
                    #loss变量初始化
                    epoch_recon_loss, epoch_kl_loss = 0, 0
                    nmi_all, ari_all = [], []
                    x = features.float().to(device)
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()

                    #使用loss函数进行loss计算
                    loss, Adj, recon_loss, kl_loss = loss_gcl(x, model, graph_learner, features, anchor_adj, alpha, beta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 10)
                    optimizer_cl.step()
                    optimizer_learner.step()

                    epoch_kl_loss += kl_loss.item()
                    epoch_recon_loss += recon_loss.item()

                    iteration += 1

                    # Structure Bootstrapping
                    if (1 - args.tau) and (args.c == 0 or (epoch + 1) % args.c == 0):
                        if args.sparse:
                            learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                            anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                      + learned_adj_torch_sparse * (1 - args.tau)
                            anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                        else:
                            anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                    if epoch % args.eval_freq == 0:
                        if args.downstream_task == 'clustering':
                            model.eval()
                            graph_learner.eval()
                            #embedding后的数据在此记录
                            _, _, embedding, _ = model(features, Adj)
                            embedding = embedding.cpu().detach().numpy()

                            kmeans = KMeans(n_clusters=nclasses, n_init=20, random_state=0).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            adata.obs['kmeans'] = predict_labels
                            cm, ari, nmi = cluster_report(adata.obs[reference].cat.codes, predict_labels)
                            nmi_all.append(nmi)
                            ari_all.append(ari)

                            wandb.log({"loss": loss.item(), "NMI": nmi, "ARI": ari, "recon_loss": recon_loss/len(x), "kl_loss": kl_loss/len(x)})
                
                    epoch_i += 1
                    tq.set_postfix_str('recon_loss {:.3f} kl_loss {:.3f}'.format(
                    epoch_recon_loss/(epoch + 1), epoch_kl_loss/(epoch + 1)))
                wandb.finish()
                if args.downstream_task == 'clustering':
                    print('SUBLIME混淆矩阵为')
                    print(cm)
                    print("Final NMI: ", nmi)
                    print("Final ARI: ", ari)
            if outdir:
                torch.save(model.state_dict(),os.path.join(outdir, 'scDGEC_{}_model.pt'.format(dataset)))  # save model
                torch.save(graph_learner.state_dict(), os.path.join(outdir, 'scDGEC_{}_graphlearner.pt'.format(dataset)))
        else:
            print('\n## Loading Model: {}\n'.format(pretrain_model))
            print('\n## Loading Model: {}\n'.format(pretrain_graphlearner))
            model.load_model(pretrain_model)
            graph_learner.load_model(pretrain_graphlearner)
            model.to(device)

    adata.obsm['latent'] = model.encodeBatch(features, Adj, device=device, out='z')
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent_nmi')
   # 2. cluster

    sc.set_figure_params(dpi=80, figsize=(6,6), fontsize=10)
    if outdir:
        sc.settings.figdir = outdir
        #这里需要修改名字
        save = 'scDGEC_{}.png'.format(dataset)
    else:
        save = None
    if  embed == 'UMAP':
        sc.tl.umap(adata, min_dist=0.1)
        color = [c for c in ['celltype',  'kmeans_nmi', 'leiden', 'cell_type'] if c in adata.obs]
        sc.pl.umap(adata, color=color, save=save, show=False, wspace=0.4, ncols=4)
    elif  embed == 'tSNE':
        sc.tl.tsne(adata, use_rep='latent')
        color = [c for c in ['celltype',  'kmeans_nmi', 'leiden', 'cell_type'] if c in adata.obs]
        sc.pl.tsne(adata, color=color, save=save, show=False, wspace=0.4, ncols=4)
    
    if  impute:
        print("Imputation")
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
        adata.obsm['binary'] = binarization(adata.obsm['impute'], adata.X)
    
    if outdir:
        #这里需要修改名字
        adata.write(outdir+'scDGEC_{}_adata.h5ad'.format(dataset), compression='gzip')
    
    return adata