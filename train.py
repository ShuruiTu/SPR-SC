import os.path
import sys

from model import get_model
from copy import deepcopy
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict, Counter
from motif import extract_motif_features
import pickle
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
epsilon = 1e-8
from baselines import ecc, community
from tqdm import tqdm
from telemetry import candidate_diagnostics, classifier_diagnostics

def _metric_record(reconstructed, ground_truth):
    precision, recall, f1, jaccard = get_performance_wrt_ground_truth(reconstructed, ground_truth)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'jaccard': jaccard}


def _size_group(size):
    """Use fixed bins so Fig. 11 and Fig. 12 share identical axes."""
    return str(size) if size <= 4 else '5+'


def _size_stratified_metrics(outputs, ground_truth):
    """Exact-match metrics within each true/predicted hyperedge-size bin."""
    groups = sorted({_size_group(len(edge)) for edge in ground_truth},
                    key=lambda value: (value == '5+', value))
    result = {}
    for method, output in outputs.items():
        result[method] = {}
        for group in groups:
            truth = {edge for edge in ground_truth if _size_group(len(edge)) == group}
            predicted = {edge for edge in output if _size_group(len(edge)) == group}
            metrics = _metric_record(predicted, truth)
            metrics['truth_count'] = len(truth)
            metrics['reconstructed_count'] = len(predicted)
            result[method][group] = metrics
    return result


def _positive_scores(model, features):
    """Return P(y=1), including sklearn's single-class fitted case."""
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(features.shape[0], dtype=float)
    probabilities = model.predict_proba(features)
    return probabilities[:, classes.index(1)]


def _binary_f1(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _fit_with_threshold(model, features, labels, args, logger, candidate_group):
    """Fit one candidate classifier and optionally tune its threshold."""
    fallback = args.decision_threshold
    if not features.shape[0]:
        return model, fallback
    if args.decision_threshold_mode == 'fixed':
        model.fit(features, labels)
        return model, fallback

    labels = np.asarray(labels)
    class_counts = Counter(labels.tolist())
    if len(class_counts) < 2 or min(class_counts.values()) < 2 or len(labels) < 10:
        logger.warning(
            '%s threshold validation unavailable for label counts %s; using %.3f.',
            candidate_group, dict(class_counts), fallback)
        model.fit(features, labels)
        return model, fallback

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        features, labels, test_size=args.threshold_validation_fraction,
        random_state=args.seed, stratify=labels)
    model.fit(X_fit, y_fit)
    scores = _positive_scores(model, X_validation)
    thresholds = np.unique(np.r_[np.arange(0.05, 1.0, 0.05), fallback])
    ranked = [
        (_binary_f1(y_validation, scores >= threshold),
         -abs(threshold - fallback), threshold)
        for threshold in thresholds
    ]
    _, _, selected = max(ranked)
    logger.info('%s validation threshold %.3f (fallback %.3f, validation rows %d).',
                candidate_group, selected, fallback, len(y_validation))
    # Threshold selection uses only the held-out split; the final estimator can
    # then use every training candidate without touching test labels.
    model.fit(features, labels)
    return model, float(selected)


def train(dataloader, args, logger):
    model1 = get_model(args, dataloader)
    model2 = deepcopy(model1)
    (X_train1, y_train1), (X_train2, y_train2) = dataloader.split_train()
    model1, threshold1 = _fit_with_threshold(
        model1, X_train1, y_train1, args, logger, 'max-clique')
    print(X_train2.shape, y_train2.shape, y_train2.sum())
    model2, threshold2 = _fit_with_threshold(
        model2, X_train2, y_train2, args, logger, 'nested-clique')
    return evaluate((model1, model2), dataloader, args, logger,
                    thresholds=(threshold1, threshold2))


def evaluate(models, dataloader, args, logger, thresholds=None):
    model1, model2 = models if len(models) > 1 else (models[0], models[0])
    thresholds = thresholds or (args.decision_threshold, args.decision_threshold)
    X_test = dataloader.X_test
    num_max_cliques_test = dataloader.get_num_max_candidates('test')
    def predict_or_reject(model, features, candidate_group, threshold):
        """Reject a group when its corresponding classifier has no train rows.

        Strong channels can leave one candidate class (typically nested cliques)
        empty after matched channel corruption.  A sklearn MLP is then rightly
        unfitted; treating those unseen candidates as negatives is preferable to
        aborting a complete SNR sweep.
        """
        if not features.shape[0]:
            return np.array([])
        if not hasattr(model, 'classes_'):
            logger.warning('%s classifier has no training samples; rejecting %d candidates.',
                           candidate_group, features.shape[0])
            return np.zeros(features.shape[0], dtype=int)
        if args.decision_threshold_mode == 'fixed' and threshold == 0.5:
            return model.predict(features)
        return (_positive_scores(model, features) >= threshold).astype(int)

    y_hat_test1 = predict_or_reject(
        model1, X_test[:num_max_cliques_test], 'max-clique', thresholds[0])
    y_hat_test2 = predict_or_reject(
        model2, X_test[num_max_cliques_test:], 'nested-clique', thresholds[1])
    y_hat_test = np.hstack((y_hat_test1, y_hat_test2))
    truth = dataloader.graphs['simplicies_test']
    reconstructed_cliques = set(clique for pred, clique in zip(y_hat_test, dataloader.cliques['final_cliques_test']) if pred > 0.5)
    shyre = _metric_record(reconstructed_cliques, truth)
    logger.info('Our Performance: precision {:.4f}, recall {:.4f}, f1 {:.4f} jaccard {:.4f}'.format(
        shyre['precision'], shyre['recall'], shyre['f1'], shyre['jaccard']))
    sampler_tag = ('-fast' if args.candidate_generator == 'shyre_fast' else
                   '-channel-aware' if args.candidate_generator == 'shyre_channel_aware' else '')
    train_tag = '-matched-train' if args.train_channel_matched else ''
    threshold_tag = '-adaptive-threshold' if args.decision_threshold_mode == 'validation' else ''
    multi_train_tag = ('-multi-train' if
                       args.train_channel_replicates * len(args.train_snr_offsets) > 1 else '')
    variant = 'SHyRe{}{}{}{}{}-{}'.format(
        sampler_tag, train_tag, '-soft' if args.soft_reliability else '',
        threshold_tag, multi_train_tag, args.features)
    logger.info('Model variant: %s', variant)
    outcome = {
        'performance': {'SHyRe': shyre},
        'decision_thresholds': {
            'mode': args.decision_threshold_mode,
            'max_clique': thresholds[0],
            'nested_clique': thresholds[1],
        },
        '_reconstructed_cliques': reconstructed_cliques,
    }

    if args.enable_candidate_metrics:
        test_candidates = dataloader.cliques['final_cliques_test']
        test_split = dataloader.get_num_max_candidates('test')
        train_candidates = dataloader.cliques['final_cliques_train']
        train_split = dataloader.get_num_max_candidates('train')
        outcome['candidate'] = candidate_diagnostics(
            test_candidates, truth, test_split, dataloader.y_test)
        outcome['candidate']['training'] = candidate_diagnostics(
            train_candidates, dataloader.graphs['simplicies_train'],
            train_split, dataloader.y_train)
        outcome['candidate']['classifier'] = classifier_diagnostics(
            y_hat_test, test_candidates, truth, test_split)
        logger.info('Candidate Metrics: %s', outcome['candidate'])

    # Bayesian-MDL depends on graph-tool and remains separately disabled.
    logger.info('Baseline: Bayesian-MDL skipped (disabled for this runtime).')
    baseline_outputs = {
        'Max Clique': dataloader.cliques['max_cliques_test'],
        'ECC': ecc.get_edge_clique_cover(dataloader.graphs['G_test']),
        'DEMON': community.get_demon_communities(dataloader.graphs['G_test']),
    }
    for name, output in baseline_outputs.items():
        metric = _metric_record(output, truth)
        outcome['performance'][name] = metric
        logger.info('Baseline: {} precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(
            'Demon' if name == 'DEMON' else name, metric['precision'], metric['recall'], metric['f1'], metric['jaccard']))

    if args.enable_cfinder:
        communities_kclique, best_k = community.get_kclique_communities(
            dataloader.graphs['G_test'], dataloader.graphs['G_test'], dataloader.graphs['simplicies_train'])
        metric = _metric_record(communities_kclique, truth)
        outcome['performance']['CFinder'] = metric
        logger.info('Baseline: CFinder (k={}) precision {:.4f}, recall {:.4f}, f1 {:.4f}, jaccard {:.4f} '.format(
            best_k, metric['precision'], metric['recall'], metric['f1'], metric['jaccard']))
    else:
        logger.info('Baseline: CFinder skipped (disabled for channel-SNR benchmark).')
    if args.enable_size_stratified_metrics:
        outputs = {'SHyRe': reconstructed_cliques}
        outputs.update(baseline_outputs)
        if args.enable_cfinder:
            outputs['CFinder'] = communities_kclique
        outcome['size_stratified'] = _size_stratified_metrics(outputs, truth)
        logger.info('Size-stratified metrics recorded for groups: %s',
                    sorted(next(iter(outcome['size_stratified'].values())).keys()))
    return outcome


def get_performance_wrt_ground_truth(reconstructed, ground_truth):
    correct_cliques = reconstructed & ground_truth
    precision = len(correct_cliques) / len(reconstructed) if len(reconstructed)>0 else 0
    recall = len(correct_cliques) / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if precision * recall > 0 else 0
    jaccard = len(correct_cliques) / len(reconstructed | ground_truth)
    return precision, recall, f1, jaccard


class DataLoader:
    def __init__(self, graphs, cliques, args, logger):
        self.args = args
        self.graphs = graphs
        self.cliques = cliques
        self.logger = logger
        self.feature_names = ['clique_type', 'clique_size', 'edge_degree_all_dup', 'edge_degree_mean',
                              'node_degree_mean', 'node_degree_mean_recur', 'node_degree_mean2', 'cluster_coef_mean', 'parent_cliques']
        if args.soft_reliability:
            self.feature_names += ['reliability_mean', 'reliability_min', 'reliability_geomean', 'reliability_logit_mean', 'reliability_below_tau']
        self.node_features, self.max_candidates = {}, {}
        self.X_train, self.cliques['final_cliques_train'], self.feature_dict_train, self.node_features['train'] = self.extract_features('train')
        self.X_test, self.cliques['final_cliques_test'], self.feature_dict_test, self.node_features['test'] = self.extract_features('test')
        self.y_train = self.create_labels(mode='train')
        self.y_test = self.create_labels(mode='test')
        self.k_best = 10

        if self.k_best > 0 and self.args.features == 'motif':
            self.X_train, self.X_test = self.select_k_best(self.X_train, self.y_train, self.X_test)
        self.logger.info('test max cliques {}, nested cliques {}'.format(self.get_num_max_candidates('test'),
                                                               len(self.y_test) - self.get_num_max_candidates('test')))

        self.logger.info('test real max cliques in search universe: {}'.format(self.y_test[:self.get_num_max_candidates('test')].sum()))
        self.logger.info('test real smaller cliques in search universe: {} {}'.format(self.y_test[self.get_num_max_candidates('test'):].sum(),
              len(set(self.cliques['final_cliques_test'][self.get_num_max_candidates('test'):]) & self.graphs['simplicies_test'])))
        if self.args.save_features == 1:
            self.save_features()

    def save_features(self):
        with open('data/{}/feature_labels.pkl'.format(self.args.dataset), 'wb') as f:
            pickle.dump({'features': self.X_test, 'label': self.y_test,
                         'feature_dict': self.feature_dict_test,
                         'feature_names': self.feature_names,
                         'cut': self.get_num_max_candidates('test')}, f)
        self.logger.info('features saved.')

    def extract_features(self, mode):
        max_cliques = self.cliques['max_cliques_{}'.format(mode)]
        child2parents = self.get_child_parents_map(mode)
        candidates_ = set(child2parents.keys())
        strict_generator = self.args.candidate_generator == 'strict_max_clique'
        if self.args.ablation > 0 or strict_generator:
            max_candidates = candidates_ & max_cliques if not strict_generator else max_cliques
        else:
            max_candidates = max_cliques
        self.max_candidates[mode] = max_candidates
        nested_candidates = candidates_ - max_candidates
        final_cliques = list(max_candidates) + list(nested_candidates)

        node_degree = self.get_node_degree(mode)
        if self.args.features == 'count':
            # Structural count features from the original SHyRe path.
            node_degree_recur = self.get_node_degree_recur(mode)
            node_degree2 = self.get_node_degree2(mode)
            edge_degree = self.get_edge_degree(mode)
            cluster_coef = self.get_cluster_coef(mode)
            feature_names = self.feature_names
            feature_dict = {feature_name: [] for feature_name in feature_names}
            for i, clique in tqdm(enumerate(final_cliques)):
                size = len(clique)
                feature_dict['clique_type'].append(int(i >= len(max_candidates)))
                feature_dict['clique_size'].append(size)
                edge_degrees = [edge_degree[edge] for edge in combinations(clique, 2)]
                feature_dict['edge_degree_all_dup'].append(all(d>1 for d in edge_degrees) if len(clique) > 1 else False)
                feature_dict['edge_degree_mean'].append(sum(edge_degrees) / max([(size*(size-1)/2), 1])) #
                feature_dict['node_degree_mean'].append(sum(node_degree[node] for node in clique) / size)
                feature_dict['node_degree_mean_recur'].append(sum(node_degree_recur[node] for node in clique) / size)
                feature_dict['node_degree_mean2'].append(sum(node_degree2[node] for node in clique)/size)
                if self.args.soft_reliability:
                    reliability = self.graphs.get('edge_reliability_{}'.format(mode), {})
                    probabilities = [reliability.get(tuple(sorted(edge)), 0.5) for edge in combinations(clique, 2)] or [1.0]
                    probabilities = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)
                    feature_dict['reliability_mean'].append(probabilities.mean())
                    feature_dict['reliability_min'].append(probabilities.min())
                    feature_dict['reliability_geomean'].append(np.exp(np.log(probabilities).mean()))
                    feature_dict['reliability_logit_mean'].append(np.log(probabilities / (1.0 - probabilities)).mean())
                    feature_dict['reliability_below_tau'].append((probabilities < self.args.channel_tau_e).mean())
                feature_dict['cluster_coef_mean'].append(sum(cluster_coef[node] for node in clique)/size)
                feature_dict['parent_cliques'].append(len(child2parents[clique]) if clique in child2parents else 0)


            # NumPy 2 removed np.float; builtin float is the same dtype here.
            X = np.zeros(shape=(len(final_cliques), len(feature_dict)), dtype=float)
            for i, feature_name in enumerate(feature_names):
                X[:, i] = feature_dict[feature_name]

        else:
            # motif based features
            X, feature_dict = extract_motif_features(candidates=final_cliques,
                                                     H=self.cliques['max_cliques_{}'.format(mode)],
                                                     node_degree=node_degree, use_c=self.args.use_c,
                                                     jobs=self.args.jobs, args=self.args)
            if self.args.soft_reliability:
                reliability = self.graphs.get('edge_reliability_{}'.format(mode), {})
                rows = []
                for clique in final_cliques:
                    p = [reliability.get(tuple(sorted(edge)), .5) for edge in combinations(clique, 2)] or [1.0]
                    p = np.clip(np.asarray(p), epsilon, 1-epsilon)
                    rows.append([p.mean(), p.min(), np.exp(np.log(p).mean()), np.log(p/(1-p)).mean(), (p < self.args.channel_tau_e).mean()])
                X = np.hstack((X, np.asarray(rows)))
        if self.args.ext:
            X = self.extend_features(X, final_cliques, child2parents, mode)
        split = self.get_num_max_candidates(mode)
        X= (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + epsilon)  # normalize
        node_degree_arr = np.zeros(len(node_degree))
        for i in range(len(node_degree_arr)):
            node_degree_arr[i] = node_degree[i]
        # The channel adapter preserves a shared node universe by adding
        # isolates.  Sparse received graphs can therefore have a zero 20th
        # percentile; keep a degree-zero bucket instead of constructing a
        # zero-width one-hot matrix and indexing it with -1.
        one_hot_dim = max(1, int(np.quantile(node_degree_arr, 0.2) if mode == 'train' else self.node_features['train'].shape[-1]))
        node_degree_arr = np.clip(node_degree_arr, a_min=None, a_max=one_hot_dim-1).astype(int)
        node_features = np.eye(one_hot_dim)[node_degree_arr]

        return X, final_cliques, feature_dict, node_features

    def get_child_parents_map(self, mode):
        child2parents = defaultdict(list)  # {child:[parent1, parent2, ...], child2: }
        count = 0
        for parent, children in self.cliques['children_cliques_{}'.format(mode)].items():
            n_children = len(children)
            count += n_children
            for child in children:
                child2parents[child].append(parent)
        self.logger.info('{}: {} unique candidates out of {} sampled candidates'.format(mode, len(child2parents), count))

        return child2parents

    def get_edge_degree(self, mode):
        edge2count = {}
        for clique in self.cliques['max_cliques_{}'.format(mode)]:
            for edge in combinations(clique, 2):
                edge_ = tuple(sorted(edge))
                if edge_ not in edge2count:
                    edge2count[edge_] = 1
                else:
                    edge2count[edge_] += 1
        return edge2count

    def get_node_degree(self, mode):
        return self.graphs['G_{}'.format(mode)].degree

    def get_node_degree_recur(self, mode):
        node_degree = self.get_node_degree(mode)
        node_degree_recur = {}
        for node in self.graphs['G_{}'.format(mode)].nodes:
            neighbor_degrees = [node_degree[neighbor] for neighbor in self.graphs['G_{}'.format(mode)].neighbors(node)]
            node_degree_recur[node] = np.array(neighbor_degrees).mean() if len(neighbor_degrees) > 0 else 0.0
        return node_degree_recur

    def get_node_degree2(self, mode):
        node2count = {}
        for clique in self.cliques['max_cliques_{}'.format(mode)]:
            for node in clique:
                if node not in node2count:
                    node2count[node] = 1
                else:
                    node2count[node] += 1
        return node2count

    def get_cluster_coef(self, mode):
        cluster_coef = nx.algorithms.cluster.clustering(self.graphs['G_{}'.format(mode)])
        return cluster_coef

    def create_labels(self, mode):
        Y = [int(clique in self.graphs['simplicies_{}'.format(mode)]) \
                  for clique in self.cliques['final_cliques_{}'.format(mode)]]
        Y = np.array(Y)
        return Y

    def get_preliminary_results(self, mode):
        if mode == 'train':
            Y = self.y_train
        else:
            Y = self.y_test
        precision = Y.sum() / len(Y)
        recall = Y.sum() / len(self.graphs['simplicies_{}'.format(mode)])
        return precision, recall

    def get_num_max_candidates(self, mode):
        if mode == 'train' and hasattr(self, '_augmented_train_max_count'):
            return self._augmented_train_max_count
        return len(self.max_candidates[mode])

    def get_num_final_cliques(self, mode):
        return len(self.cliques['final_cliques_{}'.format(mode)])

    def get_num_nodes(self, mode):
        return len(self.graphs['G_{}'.format(mode)])

    def save_features_to_csv(self):
        a = np.concatenate([self.y_test[:, None], self.X_test], axis=1)
        np.savetxt("tmp/features.csv", a, fmt='%.3f', delimiter=",")

    def split_train(self):
        num_max_cliques_train = self.get_num_max_candidates('train')
        X_train1, y_train1 = self.X_train[:num_max_cliques_train], self.y_train[:num_max_cliques_train]
        X_train2, y_train2 = self.X_train[num_max_cliques_train:], self.y_train[num_max_cliques_train:]
        upsampled = False
        if upsampled:
            return self.upsample(X_train1, y_train1), self.upsample(X_train2, y_train2)
        else:
            return (X_train1, y_train1), (X_train2, y_train2)

    def augment_training(self, additional_loaders):
        """Stack independent noisy training projections by candidate type."""
        loaders = [self] + list(additional_loaders)
        max_features, max_labels, max_cliques = [], [], []
        nested_features, nested_labels, nested_cliques = [], [], []
        for loader in loaders:
            split = loader.get_num_max_candidates('train')
            max_features.append(loader.X_train[:split])
            max_labels.append(loader.y_train[:split])
            max_cliques.extend(loader.cliques['final_cliques_train'][:split])
            nested_features.append(loader.X_train[split:])
            nested_labels.append(loader.y_train[split:])
            nested_cliques.extend(loader.cliques['final_cliques_train'][split:])

        def stack_rows(parts, columns):
            nonempty = [part for part in parts if part.shape[0]]
            return np.vstack(nonempty) if nonempty else np.empty((0, columns))

        def stack_labels(parts):
            nonempty = [part for part in parts if part.shape[0]]
            return np.hstack(nonempty) if nonempty else np.empty((0,), dtype=int)

        columns = self.X_train.shape[1]
        max_X = stack_rows(max_features, columns)
        nested_X = stack_rows(nested_features, columns)
        self.X_train = np.vstack((max_X, nested_X))
        self.y_train = np.hstack((stack_labels(max_labels), stack_labels(nested_labels)))
        self.cliques['final_cliques_train'] = max_cliques + nested_cliques
        self._augmented_train_max_count = len(max_cliques)
        self.logger.info(
            'Multi-instance training merged %d projections: max candidates %d, nested candidates %d.',
            len(loaders), len(max_cliques), len(nested_cliques))

    def extend_features(self, X, final_cliques, child2parents, mode):
        X_ext = deepcopy(X)
        start = self.get_num_max_candidates(mode)
        parent2features = {k: v for k, v in zip(final_cliques, X[:start])}
        for i in range(start, X.shape[0]):
            X_ext[i] = np.array([parent2features[parent] for parent in child2parents[final_cliques[i]]]).mean(axis=0)
        return np.concatenate((X_ext, X), axis=1)

    def upsample(self, X, y):
        if 2*y.sum() >= len(y):
            return X, y
        X_upsampled, y_upsampled = resample(X[y==1], y[y==1], n_samples=int((y==0).sum()/3), replace=True,
                                            random_state=self.args.seed)
        return np.vstack((X_upsampled, X[y==0])), np.hstack((y_upsampled, y[y==0]))

    def select_k_best(self, X_train, y_train, X_test):
        s = SelectKBest(k=self.k_best)
        X_train = np.hstack((X_train[:, :9], s.fit_transform(X_train[:, 9:], y_train)))
        X_test = np.hstack((X_test[:, :9], s.transform(X_test[:,9:])))

        return X_train, X_test


def get_reconstruction_properties(reconstruction):
    lengths = np.array([len(h) for h in reconstruction])
    c = Counter([n for h in reconstruction for n in h])
    avg_deg = np.array(c.values()).mean()
    return [len(reconstruction), lengths.mean(), lengths.std(), avg_deg]
