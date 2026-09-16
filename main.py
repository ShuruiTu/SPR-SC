from utils import *
from cliques import *
from train import *
from channel import clean_reliability, posterior_candidate_graph, simulate_channel
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
    clean_training_projection = graphs['G_train'].copy()

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
    training_observations = []
    if args.train_channel_matched:
        if args.channel == 'clean':
            logger.info('Matched training channel requested with clean test graph; training graph remains clean.')
        else:
            offsets = list(args.train_snr_offsets)
            if 0.0 in offsets:
                offsets = [0.0] + [offset for offset in offsets if offset != 0.0]
            instance_index = 0
            for offset in offsets:
                for replicate in range(args.train_channel_replicates):
                    train_seed = args.channel_seed + 1 + instance_index
                    train_snr = args.snr_db + offset
                    observation = simulate_channel(
                        clean_training_projection, train_snr, args.channel,
                        tau_e=args.channel_tau_e, seed=train_seed,
                        max_background_pairs=args.max_background_pairs,
                        beta=args.channel_beta, temperature=args.channel_temperature)
                    training_observations.append({
                        'observation': observation,
                        'snr_db': train_snr,
                        'snr_offset': offset,
                        'seed': train_seed,
                        'replicate': replicate,
                    })
                    instance_index += 1
            train_observation = training_observations[0]['observation']
            graphs['G_train'] = train_observation.graph
            telemetry['training_channel_instances'] = [
                {
                    'snr_db': item['snr_db'],
                    'snr_offset': item['snr_offset'],
                    'seed': item['seed'],
                    'replicate': item['replicate'],
                    **item['observation'].metadata,
                }
                for item in training_observations
            ]
            logger.info(
                'Matched training channel %s: %d projection instance(s); primary %.1f dB has %d edges.',
                args.channel, len(training_observations), training_observations[0]['snr_db'],
                graphs['G_train'].number_of_edges())

    if args.soft_reliability:
        if args.channel == 'clean':
            graphs['edge_reliability_train'] = clean_reliability(graphs['G_train'])
            graphs['edge_reliability_test'] = clean_reliability(graphs['G_test'])
        else:
            graphs['edge_reliability_test'] = test_observation.reliability
            if train_observation is None:
                train_observation = simulate_channel(
                    clean_training_projection, args.snr_db, args.channel, tau_e=args.channel_tau_e,
                    seed=args.channel_seed + 1, max_background_pairs=args.max_background_pairs,
                    beta=args.channel_beta, temperature=args.channel_temperature)
            graphs['edge_reliability_train'] = train_observation.reliability

    if args.candidate_generator == 'shyre_channel_aware':
        if args.channel == 'clean':
            logger.info('Channel-aware candidates requested for clean graphs; no soft edges added.')
        else:
            graphs['G_candidate_test'], candidate_test_metadata = posterior_candidate_graph(
                graphs['G_test'], test_observation.reliability,
                args.channel_candidate_tau, args.channel_candidate_max_extra_edges)
            if train_observation is not None:
                train_reliability = train_observation.reliability
            else:
                train_reliability = clean_reliability(graphs['G_train'])
            graphs['G_candidate_train'], candidate_train_metadata = posterior_candidate_graph(
                graphs['G_train'], train_reliability,
                args.channel_candidate_tau, args.channel_candidate_max_extra_edges)
            telemetry['candidate_graph_test'] = candidate_test_metadata
            telemetry['candidate_graph_train'] = candidate_train_metadata
            logger.info(
                'Channel-aware candidate graphs: test +%d edges, train +%d edges (tau %.3f).',
                candidate_test_metadata['candidate_soft_edges_added'],
                candidate_train_metadata['candidate_soft_edges_added'],
                args.channel_candidate_tau)

    cliques = compute_cliques(graphs, args, logger)
    dataloader = DataLoader(graphs, cliques, args, logger)
    if len(training_observations) > 1:
        additional_loaders = []
        for item in training_observations[1:]:
            observation = item['observation']
            extra_graphs = dict(graphs)
            extra_graphs['G_train'] = observation.graph
            extra_graphs.pop('G_candidate_train', None)
            if args.soft_reliability:
                extra_graphs['edge_reliability_train'] = observation.reliability
            if args.candidate_generator == 'shyre_channel_aware':
                extra_graphs['G_candidate_train'], _ = posterior_candidate_graph(
                    observation.graph, observation.reliability,
                    args.channel_candidate_tau, args.channel_candidate_max_extra_edges)
            extra_cliques = dict(cliques)
            extra_cliques.update(compute_training_cliques(extra_graphs, args, logger))
            additional_loaders.append(DataLoader(extra_graphs, extra_cliques, args, logger))
        dataloader.augment_training(additional_loaders)
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
            'model_variant': 'SHyRe{}{}{}{}{}-{}'.format(
                ('-fast' if args.candidate_generator == 'shyre_fast' else
                 '-channel-aware' if args.candidate_generator == 'shyre_channel_aware' else ''),
                '-matched-train' if args.train_channel_matched else '',
                '-soft' if args.soft_reliability else '',
                '-adaptive-threshold' if args.decision_threshold_mode == 'validation' else '',
                '-multi-train' if len(training_observations) > 1 else '',
                args.features),
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
                'decision_threshold_mode': args.decision_threshold_mode,
                'train_channel_replicates': args.train_channel_replicates,
                'train_snr_offsets': args.train_snr_offsets,
            },
            'projection': telemetry,
            'outcome': outcome,
        }
        write_metrics_jsonl(args.metrics_jsonl, payload)
        logger.info('Metrics JSONL written to %s', args.metrics_jsonl)
