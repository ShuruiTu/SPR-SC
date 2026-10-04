from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
eps = 1e-4


def get_model(args, dataloader):
    if args.model == 'rf':
        model1 = RandomForestClassifier(
            random_state=args.seed, n_jobs=-1, verbose=0,
            n_estimators=args.rf_n_estimators,
            max_depth=None if args.rf_max_depth == 0 else args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=None if args.rf_max_features == 'all' else args.rf_max_features,
            class_weight=None if args.rf_class_weight == 'none' else args.rf_class_weight)
    elif args.model == 'mlp':
        model1 = MLPClassifier(
            random_state=args.seed, verbose=False, max_iter=args.epochs,
            hidden_layer_sizes=tuple(args.mlp_hidden_layers), alpha=args.mlp_alpha,
            learning_rate_init=args.lr, early_stopping=args.mlp_early_stopping,
            validation_fraction=args.mlp_validation_fraction,
            n_iter_no_change=args.mlp_n_iter_no_change)
    elif args.model == 'lr':
        model1 = LogisticRegression(
            random_state=args.seed, n_jobs=-1, C=args.logistic_c,
            max_iter=args.logistic_max_iter,
            class_weight=None if args.logistic_class_weight == 'none'
            else args.logistic_class_weight)
    else:
        raise Exception
    return model1
