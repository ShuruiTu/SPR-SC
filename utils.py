import argparse
import networkx as nx
from itertools import combinations
import numpy as np
import random
import os
import sys
import logging
import time


def set_up():
    args, sys_argv = get_args()
    logger = set_up_logger(args, sys_argv)
    set_random_seed(args.seed)
    return args, logger


def get_args():
    parser = argparse.ArgumentParser('Interface for hypergraphs reconstruction framework')

    # key parameters: dataset, features, model, and sampling budget
    parser.add_argument('--dataset', '--d', type=str, default='dblp', help='dataset name')
    parser.add_argument('--beta', type=int, default=1e6, help='sampling budget')
    parser.add_argument('--features', type=str, default='count', help='type of features used to characterize structural property, heuristic or motif-based')
    parser.add_argument('--model', choices=['mlp', 'lr', 'rf'], default='mlp', help='ML model to use')
    parser.add_argument('--class_balance', '--class-balance',
                        choices=['none', 'upsample'], default='none',
                        help='optional label balancing applied independently by candidate type')
    parser.add_argument('--upsample_positive_ratio', '--upsample-positive-ratio',
                        type=float, default=1.0 / 3.0,
                        help='target positive/negative ratio for upsampling')
    # parser.add_argument('--gnn_model', type=str, default='GIN', help='GNN model to use, valid only when model=gnn')

    # moderate training process
    parser.add_argument('--epochs', type=int, default=2000, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='random seed')

    # more experiments
    parser.add_argument('--setting', type=str, default='f', help='fully supervised (f) or semi-supervised with 10% labels (s)')
    parser.add_argument('--label_rate', type=float, default=0.1, help='fraction of labeling in semi-supervised learning')
    parser.add_argument('--ablation', type=int, default=0, help='alternative ways to sample cliques, for ablation study')

    # parameters that have no effect on results
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--use_c', type=int, default=1, help='whether to use c++ code for extracting motif')
    parser.add_argument('--jobs', type=int, default=1, help='number of cores to use for extracting motifs')
    parser.add_argument('--save_features', type=int, default=0, help='epochs')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--log_dir', type=str, default='log', help='log root directory')
    parser.add_argument('--downsample', type=int, default=0, help='downsample neighborhood in motif extraction')

    # deprecated

    parser.add_argument('--max_child_size', type=int, default=1, help='maximum size of the child')
    parser.add_argument('--ext', type=int, default=0, help='whether to extend features')
    # Optional, independent modules for Fig. 3/4/9/10 and ablation studies.
    parser.add_argument('--enable_projection_retention', action='store_true', help='retain a random alpha fraction of test projection edges before the channel')
    parser.add_argument('--projection_retention', type=float, default=1.0, help='alpha in [0, 1], effective only with --enable_projection_retention')
    parser.add_argument('--retention_seed', type=int, default=None, help='RNG seed for projection retention; defaults to channel_seed')
    parser.add_argument('--enable_candidate_metrics', action='store_true', help='record candidate coverage statistics for Fig. 4')
    parser.add_argument('--enable_storage_metrics', action='store_true', help='record received/reconstructed storage statistics for Fig. 10')
    parser.add_argument('--candidate_generator', choices=['shyre', 'shyre_fast', 'shyre_channel_aware', 'strict_max_clique', 'random', 'head', 'tail'], default='shyre', help='independent candidate-generation module for ablations and Fig. 4')
    parser.add_argument('--enable_runtime_metrics', action='store_true', help='record wall-clock and peak RSS telemetry for scaling studies')
    parser.add_argument('--enable_size_stratified_metrics', action='store_true', help='record per-hyperedge-size F1 for Fig. 11/12')
    parser.add_argument('--metrics_jsonl', type=str, default='', help='optional JSONL destination for enabled telemetry modules')
    parser.add_argument('--channel', choices=['clean', 'awgn', 'rayleigh'], default='clean', help='test projection channel')
    parser.add_argument('--snr_db', type=float, default=60.0, help='channel SNR in dB')
    parser.add_argument('--channel_seed', type=int, default=123, help='received-graph RNG seed')
    parser.add_argument('--max_background_pairs', type=int, default=10000, help='sampled non-edges for channel false positives')
    parser.add_argument('--channel_tau_e', type=float, default=0.58, help='posterior hard-edge threshold')
    parser.add_argument('--channel_beta', type=float, default=1.0, help='channel LLR scale')
    parser.add_argument('--channel_temperature', type=float, default=1.0, help='channel posterior temperature')
    parser.add_argument('--enable_cfinder', action='store_true', help='run CFinder baseline (disabled by default)')
    parser.add_argument('--soft_reliability', action='store_true', help='append channel reliability statistics to count features')
    parser.add_argument('--soft_reliability_mode', '--soft-reliability-mode',
                        choices=['off', 'basic', 'distribution'], default='off',
                        help='soft reliability feature set; basic preserves the v1.0 soft-count features')
    parser.add_argument('--train_channel_matched', action='store_true', help='apply an independently drawn channel with matching type/SNR to the training projection')
    parser.add_argument('--train_channel_replicates', '--train-channel-replicates', type=int, default=1,
                        help='independent noisy training projections per train SNR')
    parser.add_argument('--train_snr_offsets', '--train-snr-offsets', type=float, nargs='+', default=[0.0],
                        help='training SNR offsets relative to the test SNR, e.g. -2 0 2')
    parser.add_argument('--decision_threshold_mode', '--decision-threshold-mode',
                        choices=['fixed', 'validation'], default='fixed',
                        help='fixed 0.5-style decision or per-candidate-type validation threshold')
    parser.add_argument('--decision_threshold', '--decision-threshold', type=float, default=0.5,
                        help='positive-class threshold used in fixed mode and as validation fallback')
    parser.add_argument('--threshold_validation_fraction', '--threshold-validation-fraction',
                        type=float, default=0.2,
                        help='training-candidate fraction reserved only for threshold selection')
    parser.add_argument('--channel_candidate_tau', '--channel-candidate-tau', type=float, default=0.45,
                        help='posterior threshold for soft edges used only by shyre_channel_aware')
    parser.add_argument('--channel_candidate_max_extra_edges', '--channel-candidate-max-extra-edges',
                        type=int, default=2000,
                        help='maximum posterior soft edges added to each candidate-search graph')

    try:
        args = parser.parse_args()
        if args.setting == 's':
            args.dataset = args.dataset + '-s'
        if args.soft_reliability and args.soft_reliability_mode == 'off':
            args.soft_reliability_mode = 'basic'
        if args.soft_reliability_mode != 'off':
            args.soft_reliability = True
        if not 0.0 < args.decision_threshold < 1.0:
            parser.error('--decision-threshold must be in (0, 1)')
        if not 0.0 < args.threshold_validation_fraction < 0.5:
            parser.error('--threshold-validation-fraction must be in (0, 0.5)')
        if not 0.0 < args.channel_candidate_tau < 1.0:
            parser.error('--channel-candidate-tau must be in (0, 1)')
        if args.channel_candidate_max_extra_edges < 0:
            parser.error('--channel-candidate-max-extra-edges must be non-negative')
        if args.train_channel_replicates < 1:
            parser.error('--train-channel-replicates must be at least 1')
        if not 0.0 < args.upsample_positive_ratio <= 1.0:
            parser.error('--upsample-positive-ratio must be in (0, 1]')
        if ((args.train_channel_replicates != 1 or args.train_snr_offsets != [0.0])
                and not args.train_channel_matched):
            parser.error('multi-instance channel training requires --train_channel_matched')
        if ((args.train_channel_replicates != 1 or args.train_snr_offsets != [0.0])
                and args.channel == 'clean'):
            parser.error('multi-instance channel training requires awgn or rayleigh')
        if ((args.train_channel_replicates != 1 or args.train_snr_offsets != [0.0])
                and args.features != 'count'):
            parser.error('multi-instance channel training currently supports count features only')
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def load_graphs(args, logger):
    dataset_dir = args.data_dir+args.dataset+'/'
    graphs = {}
    graphs['simplicies_train'] = read_simplicies(dataset_dir, mode='train')
    graphs['simplicies_test'] = read_simplicies(dataset_dir,  mode='test')
    graphs['G_train'] = construct_graph(graphs['simplicies_train'])
    graphs['G_test'] = construct_graph(graphs['simplicies_test'])
    logger.info('Finish loading graphs.')


    logger.info('Nodes train: {}, test: {}'.format(graphs['G_train'].number_of_nodes(), graphs['G_test'].number_of_nodes()))
    logger.info('Simplicies train: {}, test: {}'.format(len(graphs['simplicies_train']), len(graphs['simplicies_test'])))
    return graphs


def read_simplicies(file_dir, mode='train'):
    simplicies = []
    with open(file_dir + '{}.txt'.format(mode), 'r') as f:
        for line in f.readlines():
            simplicies.append(tuple(sorted(set([int(node) for node in line.strip().split(' ')]))))

    try:
        assert (len(set(simplicies)) == len(simplicies)) # no duplicate
        nodes = set([node for simplex in simplicies for node in simplex])
        assert(min(nodes) == 0)
        assert (max(nodes) == len(nodes)-1)  # compact indexing

    # sanity check
    except AssertionError:
        print('Sanity check failed, reindexing the hypergraphs ...')
        all_nodes = sorted(set(n for s in simplicies for n in s))
        node2i = {node: i for i, node in enumerate(all_nodes)}
        simplicies = [tuple(sorted(set([node2i[n] for n in s]))) for s in simplicies]

    return set(simplicies)


def construct_graph(simplicies):
    G = nx.Graph()
    for s in simplicies:
        if len(s) == 1:
            G.add_node(s[0])
            continue
        for e in combinations(s, 2):
            G.add_edge(*e)
    print('number of nodes in construct graph', G.number_of_nodes())
    return G


def set_random_seed(seed):
    seed = seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_up_logger(args, sys_argv):
    # set up running log
    runtime_id = '{}-{}-{}-{}'.format(args.dataset, str(args.beta), args.features[:3], str(time.time()))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_dir = '{}/{}/'.format(args.log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_path = log_dir + runtime_id + '.log'
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger
