import numpy as np
from sklearn.datasets import make_classification

def load_or_generate_data(
    n_samples = 500,
    n_features = 2,
    n_redundant = 0,
    n_informative = 2,
    random_state = 42,
    n_clusters_per_class = 1,
):
    X, y = make_classification(
        n_samples = n_samples,
        n_features = n_features,
        n_redundant = n_redundant,
        n_informative = n_informative,
        random_state = random_state,
        n_clusters_per_class = n_clusters_per_class,
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    return X, y


def train_test_split_stratified(X, y, test_size = 0.3, random_state = 42):
    rng = np.random.default_rng(random_state)

    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    rng.shuffle(class_0_indices)
    rng.shuffle(class_1_indices)

    n_test_0 = int(len(class_0_indices) * test_size)
    n_test_1 = int(len(class_1_indices) * test_size)

    test_indices = np.concatenate([
        class_0_indices[:n_test_0],
        class_1_indices[:n_test_1],
    ])

    train_indices = np.concatenate([
        class_0_indices[n_test_0:],
        class_1_indices[n_test_1:],
    ])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def train_val_split(X_train, y_train, val_size = 0.2, random_state = 42):
    rng = np.random.default_rng(random_state)

    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    rng.shuffle(class_0_indices)
    rng.shuffle(class_1_indices)

    n_val_0 = int(len(class_0_indices) * val_size)
    n_val_1 = int(len(class_1_indices) * val_size)

    val_indices = np.concatenate([
        class_0_indices[:n_val_0],
        class_1_indices[:n_val_1],
    ])

    new_train_indices = np.concatenate([
        class_0_indices[n_val_0:],
        class_1_indices[n_val_1:],
    ])

    rng.shuffle(val_indices)
    rng.shuffle(new_train_indices)

    X_train_new = X_train[new_train_indices]
    y_train_new = y_train[new_train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]


    return X_train_new, X_val, y_train_new, y_val


def fit_standardization(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1.0

    return mean, std


def apply_standardization(X, mean, std):
    return (X - mean) / std


def standardize_train_test(X_train, X_test):
    mean, std  = fit_standardization(X_train)
    X_train_scaled = apply_standardization(X_train, mean, std)
    X_test_scaled = apply_standardization(X_test, mean, std)

    return X_train_scaled, X_test_scaled, mean, std


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)

def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp,tn,fp,fn

def precision_score(y_true, y_pred):
    tp, _,fp, _ = confusion_counts(y_true, y_pred)

    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    tp, _, _, fn = confusion_counts(y_true, y_pred)

    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def roc_curve_points(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = np.sort(np.unique(y_score))[::-1]
    thresholds = np.concatenate(([1.0], thresholds, [0.0]))

    fpr_values = []
    tpr_values = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp, tn, fp, fn = confusion_counts(y_true, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return np.array(fpr_values), np.array(tpr_values)

def roc_auc_score_manual(y_true, y_score):
    fpr, tpr = roc_curve_points(y_true, y_score)

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    return np.trapz(tpr, fpr)


    