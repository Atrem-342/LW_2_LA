from perceptron import Perceptron
from utils import (
    load_or_generate_data,
    train_test_split_stratified,
    train_val_split,
    fit_standardization,
    apply_standardization,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score_manual,
    roc_curve_points,
)
from plots import (
    plot_losses,
    plot_decision_boundary,
    plot_roc_curve, 
    plot_misclassified_points,
    )

from generate_data import generate_synthetic_data

def run_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    lr,
    batch_size,
    epochs,
    init_mode,
    random_state = 42,
):
    mean , std = fit_standardization(X_train)

    X_train_scaled = apply_standardization(X_train, mean, std)
    X_val_scaled = apply_standardization(X_val, mean, std)
    X_test_scaled = apply_standardization(X_test, mean, std)

    model = Perceptron(
        input_dim = X_train_scaled.shape[1],
        init_mode = init_mode,
        random_state = random_state,
    )

    train_losses, val_losses = model.fit(
        X_train = X_train_scaled,
        y_train = y_train,
        X_val = X_val_scaled,
        y_val = y_val,
        epochs = epochs,
        lr = lr,
        batch_size = batch_size,
    )

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    fpr, tpr = roc_curve_points(y_test, y_test_proba)
    roc_auc = roc_auc_score_manual(y_test, y_test_proba)


    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

    }
    
def choose_data():

    print("1 - load_or_generate_data()")
    print("2 - geenrate_synthetic_data(data_type = 'linear')")
    print("3 - geenrate_synthetic_data(data_type = 'XOR')")
    print("4 - geenrate_synthetic_data(data_type = 'circle')")
    experiment = int(input("Введите номер эксперимента: "))
    if experiment == 1:
        return load_or_generate_data()
    
    if experiment == 2:
        return generate_synthetic_data(data_type="linear")
    
    if experiment == 3:
        return generate_synthetic_data(data_type="xor")
    
    if experiment == 4:
        return generate_synthetic_data(data_type="circle")
    
    print("Неверный ввод. Будет использован вариант 1.")
    return load_or_generate_data()

def main():

    X, y = choose_data()

    print("Исходные данные:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Количество объектов класса 0: {(y == 0).sum()}")
    print(f"Количество объектов класса 1: {(y == 1).sum()}")
    print()

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X,
        y,
        test_size = 0.3,
        random_state = 42,
    )

    print("После train/test split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print()

    X_train_inner, X_val, y_train_inner, y_val = train_val_split(
        X_train,
        y_train,
        val_size = 0.2,
        random_state = 42,
    )

    print("После train/val split:")
    print(f"X_train_inner shape: {X_train_inner.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train_inner shape: {y_train_inner.shape}")
    print(f"y_val shape: {y_val.shape}")
    print()

    if X_train_inner.shape[1] != 2:
        raise ValueError("Ожидалось ровно 2 признака для визуализации границы.")
    
    result = run_experiment(
        X_train = X_train_inner,
        y_train = y_train_inner,
        X_val = X_val,
        y_val = y_val,
        X_test = X_test,
        y_test = y_test,
        lr = 0.1,
        batch_size = 64,
        epochs = 100,
        init_mode = "random_small",
        random_state = 42,
    )

    print("Первые 5 значений train loss:")
    print(result["train_losses"][:5])
    print()

    print("Первые 5 значений val loss:")
    print(result["val_losses"][:5])
    print()

    print("Результаты обучения:")
    print(f"Train accuracy: {result['train_acc']:.4f}")
    print(f"Test accuracy: {result['test_acc']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-score: {result['f1']:.4f}")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"Итоговые веса: {result['model'].w}")
    print(f"Итоговое смещение: {result['model'].b:.4f}")


    plot_losses(result["train_losses"], result["val_losses"])
    plot_decision_boundary(result["model"], result["X_test_scaled"], y_test)
    plot_roc_curve(result['fpr'], result['tpr'], result['roc_auc'])
    plot_misclassified_points(result['model'], result['X_test_scaled'], y_test)


if __name__ == "__main__":
    main()