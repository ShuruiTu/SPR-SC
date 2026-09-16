from utils import *
from cliques import *
from train import *
from channel import clean_reliability, simulate_channel
from time import perf_counter
from telemetry import (retain_projection_edges, runtime_metrics, storage_metrics,
                       write_metrics_jsonl)

from collections import defaultdict
def get_node2neighbors(all_sets):
    node2neighbors = defaultdict(list)
    for i, clique in enumerate(all_sets):
        for node in clique:
            node2neighbors[node].append(i)
    return node2neighbors


def get_neighbor_sets(clique, node2neighbors):
    return set([n for node in clique for n in node2neighbors[node]])


def count(all_sets, suspects=[]):
    if len(suspects) == 0:
        suspects = list(range(len(all_sets)))
    node2neighbors = get_node2neighbors(all_sets)
    pairs = []
    for suspect in suspects:
        neighbor_sets = get_neighbor_sets(all_sets[suspect], node2neighbors)
        for neighbor_set in neighbor_sets:
            if suspect == neighbor_set:
                continue
            if len(all_sets[suspect] - all_sets[neighbor_set]) == 0:
                pairs.append((suspect, neighbor_set))
                break
    return pairs


def get_data(graphs, cliques):
    max_cliques = cliques['max_cliques_test']
    edges = graphs['simplicies_test']
    suspect_sets = [set(x) for x in edges-max_cliques]
    max_sets = [set(x) for x in (edges & max_cliques)]
    return suspect_sets + max_sets, list(range(len(suspect_sets)))
if __name__ == '__main__':
    start_time = perf_counter()
    args, logger = set_up()  # args and random seed
    graphs = load_graphs(args, logger)

    # Retention acts strictly before the optional wireless channel.
    clean_projection = graphs['G_test'].copy()
    if args.enable_projection_retention:
        retention_seed = args.channel_seed if args.retention_seed is None else args.retention_seed
        transmitted_projection, retention = retain_projection_edges(
            clean_projection, args.projection_retention, retention_seed)
        logger.info('Projection retention enabled: requested alpha %.4f, edges %d -> %d',
                    args.projection_retention, clean_projection.number_of_edges(),
                    transmitted_projection.number_of_edges())
    else:
        transmitted_projection = clean_projection.copy()
        retention = {
            'projection_edges_original': clean_projection.number_of_edges(),
            'projection_edges_transmitted': clean_projection.number_of_edges(),
            'projection_retention_requested': 1.0,
            'projection_retention_actual': 1.0,
        }
    graphs['G_test'] = transmitted_projection
    telemetry = dict(retention)

    if args.channel != 'clean':
        test_observation = simulate_channel(
            transmitted_projection, args.snr_db, args.channel, tau_e=args.channel_tau_e,
            seed=args.channel_seed, max_background_pairs=args.max_background_pairs,
            beta=args.channel_beta, temperature=args.channel_temperature)
        graphs['G_test'] = test_observation.graph
        telemetry.update(test_observation.metadata)
        logger.info('Channel %s at %.1f dB: transmitted projection edges %d -> received %d',
                    args.channel, args.snr_db, transmitted_projection.number_of_edges(),
                    graphs['G_test'].number_of_edges())
    else:
        telemetry.update({
            'background_pairs_sampled': 0,
            'channel_symbol_count': 0,
            'channel_symbol_bits': 0,
            'received_projection_edges': graphs['G_test'].number_of_edges(),
        })

    train_observation = None
    if args.train_channel_matched:
        if args.channel == 'clean':
            logger.info('Matched training channel requested with clean test graph; training graph remains clean.')
        else:
            # Keep the training noise independent of the test realization while
            # matching its channel family and SNR.
            train_observation = simulate_channel(
                graphs['G_train'], args.snr_db, args.channel, tau_e=args.channel_tau_e,
                seed=args.channel_seed + 1, max_background_pairs=args.max_background_pairs,
                beta=args.channel_beta, temperature=args.channel_temperature)
            graphs['G_train'] = train_observation.graph
            logger.info('Matched training channel %s at %.1f dB: received training projection edges %d',
                        args.channel, args.snr_db, graphs['G_train'].number_of_edges())

    if args.soft_reliability:
        if args.channel == 'clean':
            graphs['edge_reliability_train'] = clean_reliability(graphs['G_train'])
            graphs['edge_reliability_test'] = clean_reliability(graphs['G_test'])
        else:
            graphs['edge_reliability_test'] = test_observation.reliability
            if train_observation is None:
                train_observation = simulate_channel(
                    graphs['G_train'], args.snr_db, args.channel, tau_e=args.channel_tau_e,
                    seed=args.channel_seed + 1, max_background_pairs=args.max_background_pairs,
                    beta=args.channel_beta, temperature=args.channel_temperature)
            graphs['edge_reliability_train'] = train_observation.reliability

    cliques = compute_cliques(graphs, args, logger)
    dataloader = DataLoader(graphs, cliques, args, logger)
    outcome = train(dataloader, args, logger)
    reconstructed = outcome.pop('_reconstructed_cliques')

    if args.enable_storage_metrics:
        outcome['storage'] = storage_metrics(graphs, reconstructed)
        logger.info('Storage Metrics: %s', outcome['storage'])
    if args.enable_runtime_metrics:
        outcome['runtime'] = runtime_metrics(start_time)
        logger.info('Runtime Metrics: %s', outcome['runtime'])
    if args.metrics_jsonl:
        payload = {
            'dataset': args.dataset,
            'channel': args.channel,
            'snr_db': args.snr_db,
            'seed': args.seed,
            'model_variant': 'SHyRe{}{}{}-{}'.format(
                '-fast' if args.candidate_generator == 'shyre_fast' else '',
                '-matched-train' if args.train_channel_matched else '',
                '-soft' if args.soft_reliability else '', args.features),
            'modules': {
                'projection_retention': args.enable_projection_retention,
                'soft_reliability': args.soft_reliability,
                'train_channel_matched': args.train_channel_matched,
                'candidate_metrics': args.enable_candidate_metrics,
                'storage_metrics': args.enable_storage_metrics,
                'runtime_metrics': args.enable_runtime_metrics,
                'size_stratified_metrics': args.enable_size_stratified_metrics,
                'cfinder': args.enable_cfinder,
                'candidate_generator': args.candidate_generator,
            },
            'projection': telemetry,
            'outcome': outcome,
        }
        write_metrics_jsonl(args.metrics_jsonl, payload)
        logger.info('Metrics JSONL written to %s', args.metrics_jsonl)
