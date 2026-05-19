import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize = (8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth = 2)
    plt.plot(epochs, val_losses, label = "Validation Loss", linewidth = 2)

    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Изменение функции потерь")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y):
    plt.figure(figsize = (8, 6))

    class_0 = y == 0
    class_1 = y == 1

    plt.scatter(
        X[class_0, 0],
        X[class_0, 1],
        label="Класс 0",
        alpha = 0.7
    )
    plt.scatter(
        X[class_1, 0],
        X[class_1, 1],
        label = "Класс 1",
        alpha = 0.7
    )

    x_min = X[:, 0].min() - 1.0
    x_max = X[:, 0].max() + 1.0
    x_values = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-8:
        y_values = -(model.w[0] * x_values + model.b) / model.w[1]
        plt.plot(
            x_values, 
            y_values, 
            color = "black", 
            linewidth = 2, 
            label = " Граница решения",
        )
    elif abs(model.w[0]) > 1e-8:
        x_boundary = -model.b / model.w[0]
        plt.axvline(x=x_boundary, color="black", linewidth=2, label="Граница решения")
        
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title("Разделяющая граница перцептрона")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize = (7, 6))
    plt.plot(fpr, tpr, linewidth = 2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайный классификатор")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_misclassified_points(model, X, y_true):
    y_pred = model.predict(X)
    misclassified = y_pred != y_true

    plt.figure(figsize = (8, 6))

    class_0 = y_true == 0
    class_1 = y_true == 1

    plt.scatter(
        X[class_0, 0],
        X[class_0, 1],
        label = "Класс 0",
        alpha = 0.6
    )
    plt.scatter(
        X[class_1, 0],
        X[class_1, 1],
        label="Класс 1",
        alpha=0.6
    )

    if np.any(misclassified):
        plt.scatter(
            X[misclassified, 0],
            X[misclassified, 1],
            facecolors = "none",
            edgecolors = "red",
            s = 120,
            linewidths = 2,
            label = "Ошибочные точки"
        )

    x_min = X[:, 0].min() - 1.0
    x_max = X[:, 0].max() + 1.0
    x_values = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-8:
        y_values = -(model.w[0] * x_values + model.b) / model.w[1]
        plt.plot(
            x_values,
            y_values,
            color="black",
            linewidth=2,
            label="Граница решения"
        )
    elif abs(model.w[0]) > 1e-8:
        x_boundary = -model.b / model.w[0]
        plt.axvline(
            x=x_boundary,
            color="black",
            linewidth=2,
            label="Граница решения"
        )

    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title("Ошибочно классифицированные точки")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()