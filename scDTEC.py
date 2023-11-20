#!/usr/bin/env python
import argparse

from scdtec import scDTEC_function


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='scDTEC:scDTEC: Unsupervised Deep Topology Embedded Characterization of Single-Cell Chromatin Accessibility Profiles')
    parser.add_argument('--data_list', '-d', type=str, default='Forebrain')
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=30)
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--pretrain_model', type=str, default=None, help='Load the trained model')
    parser.add_argument('--pretrain_graphlearner', type=str, default=None, help='Load the graphlearner model')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[1024, 128], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    parser.add_argument('--min_peaks', type=float, default=100, help='Remove low quality cells with few peaks')
    parser.add_argument('--min_cells', type=float, default=3, help='Remove low quality peaks')
    parser.add_argument('--n_feature', type=int, default=100000, help='Keep the number of highly variable peaks')#100000
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--max_iter', '-i', type=int, default=30000, help='Max iteration')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--impute', action='store_true', help='Save the imputed data in layer impute')
    parser.add_argument('--binary', action='store_true', help='Save binary imputed data in layer binary')
    parser.add_argument('--embed', type=str, default='UMAP')
    parser.add_argument('--reference', type=str, default='celltype')
    parser.add_argument('--cluster_method', type=str, default='kmeans')

    parser.add_argument('--ntrials', type=int, default=10)
    parser.add_argument('--sparse', type=int, default=0)
    parser.add_argument('--gsl_mode', type=str, default='structure_inference',
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--downstream_task', type=str, default='clustering',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    parser.add_argument('--w_decay', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=32)#512
    parser.add_argument('--rep_dim', type=int, default=16)#256
    parser.add_argument('--proj_dim', type=int, default=16) #256
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--contrast_batch_size', type=int, default=0)
    parser.add_argument('--nlayers', type=int, default=2)

    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.1)
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.8)
    parser.add_argument('--dropedge_rate', type=float, default=0.5)

    parser.add_argument('--type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('--knn_k', type=int, default=20)
    parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'], help='KNN metric method')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--activation_learner', type=str, default='relu', choices=["relu", "tanh"], help='Graph Learner activation')

    parser.add_argument('--tau', type=float, default=0.999)
    parser.add_argument('--c', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)


    args = parser.parse_args()
    adata = scDTEC_function(
        args,
        args.data_list, 
        n_centroids = args.n_centroids,
        outdir = args.outdir,
        verbose = args.verbose,
        pretrain_model = args.pretrain_model,
        pretrain_graphlearner= args.pretrain_graphlearner,
        lr = args.lr,
        batch_size = args.batch_size,
        gpu = args.gpu,
        seed = args.seed,
        encode_dim = args.encode_dim,
        decode_dim = args.decode_dim,
        latent = args.latent,
        min_peaks = args.min_peaks,
        min_cells = args.min_cells,
        n_feature = args.n_feature,
        log_transform = args.log_transform,
        max_iter = args.max_iter,
        weight_decay = args.weight_decay,
        impute = args.impute,
        binary = args.binary,
        embed = args.embed,
        reference = args.reference,
        cluster_method = args.cluster_method,
        alpha = args.alpha,
        beta = args.beta
    )

    