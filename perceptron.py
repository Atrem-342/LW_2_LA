import numpy as np

class Perceptron:
    def __init__(self, input_dim, init_mode="random_small", random_state = 42):
        self.input_dim = input_dim
        self.init_mode = init_mode
        self.random_state = random_state

        self.w = self._initialize_weights()
        self.b = 0.0

        self.train_losses = []
        self.val_losses = []

    def _initialize_weights(self):
        rng = np.random.default_rng(self.random_state)

        if self.init_mode == "zero":
            return np.zeros(self.input_dim, dtype = float)
        
        if self.init_mode == "random_small":
            return rng.normal(loc = 0.0, scale = 0.01,size = self.input_dim)
        
        if self.init_mode == "random_large":
            return rng.normal(loc = 0.0, scale = 10.0, size = self.input_dim)
        
        raise ValueError(
            "init_mode должен быть одним из: "
            "'zero', 'random_small', 'random_large'"
        )

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0/(1.0 + np.exp(-z))
    
    def forward(self, X):
        z = X @ self.w + self.b
        y_pred = self.sigmoid(z)
        return y_pred
    
    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def predict_proba(self, X):
        return self.forward(X)
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                y_pred_batch = self.forward(X_batch)

                error = y_pred_batch - y_batch

                grad_w = (X_batch.T @ error) / len(X_batch)
                grad_b = np.mean(error)

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_loss = self.compute_loss(y_train, train_pred)
            val_loss = self.compute_loss(y_val, val_pred)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        return self.train_losses, self.val_losses

            








