import numpy as np

def add_label_noise(y, noise_propability = 0.0, random_state = 42):
    rng = np.random.default_rng(random_state)

    y_noisy = y.copy()

    if noise_propability <= 0.0:
        return y_noisy
    
    flip_mask = rng.random(len(y_noisy)) < noise_propability
    print("Число перевёрнутых меток:", np.sum(flip_mask))
    y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

    return y_noisy
    
def generate_linear_data(
        n_samples = 500,
        mean_class_0 = (-2.0, -2.0),
        mean_class_1 = (2.0, 2.0),
        covariance = ((1.0, 0.0), (0.0,1.0)),
        noise_propability = 0.0,
        random_state = 42,
):
    
    rng = np.random.default_rng(random_state)

    n_class_0 = n_samples // 2
    n_class_1 = n_samples - n_class_0

    mean_class_0 = np.asarray(mean_class_0, dtype = float)
    mean_class_1 = np.asarray(mean_class_1, dtype = float)
    covariance = np.asarray(covariance, dtype = float)

    X_class_0 = rng.multivariate_normal(
        mean = mean_class_0,
        cov = covariance,
        size = n_class_0,
    )
    X_class_1 = rng.multivariate_normal(
        mean = mean_class_1,
        cov = covariance,
        size = n_class_1,
    )

    y_class_0 = np.zeros(n_class_0, dtype = int)
    y_class_1 = np.ones(n_class_1, dtype = int)

    X = np.vstack([X_class_0, X_class_1])
    y = np.concatenate([y_class_0, y_class_1])

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    y = add_label_noise(
        y,
        noise_propability = noise_propability,
        random_state = random_state,
    )

    return X, y

    
def generate_xor_data(
        n_samples = 500,
        spread = 0.5,
        noise_propability = 0.0,
        random_state = 42,
):
    rng = np.random.default_rng(random_state)

    n_base = n_samples // 4
    reminder = n_samples % 4

    cluster_sizes = [n_base, n_base, n_base, n_base]
    for i in range(reminder):
        cluster_sizes[i] += 1

    centers = [
        (-1.0,-1.0),
        (-1.0,1.0),
        (1.0,-1.0),
        (1.0,1.0),
    ]

    labels = np.array([0,1,1,0], dtype = int)

    X_parts = []
    y_parts = []    


    for (center_x, center_y), label, size in zip(centers, labels,cluster_sizes):
        cluster = rng.normal(
            loc = (center_x, center_y),
            scale = spread,
            size = (size, 2),
        )

        cluster_labels = np.full(size, label, dtype = int)

        X_parts.append(cluster)
        y_parts.append(cluster_labels)
    
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    y = add_label_noise(
        y,
        noise_propability = noise_propability,
        random_state = random_state,
    )

    return X, y
   
def generate_circle_data(
        n_samples = 500,
        inner_radius = 2.35,
        outer_radius = 3.0,
        noise_propability = 0.0,
        random_state = 42,
):
    rng = np.random.default_rng(random_state)

    X = rng.uniform(
        low = -outer_radius,
        high = outer_radius,
        size = (n_samples, 2),
    )

    distances = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

    y = (distances >= inner_radius).astype(int)

    y = add_label_noise(
        y,
        noise_propability = noise_propability,
        random_state = random_state,
    )

    return X, y
    

   
def generate_synthetic_data(
        data_type = "linear",
        n_samples = 500,
        noise_propability = 0.0,
        random_state = 42,
        **kwargs,
):
    if data_type == "linear":
        return generate_linear_data(
            n_samples = n_samples,
            noise_propability = noise_propability,
            random_state = random_state,
            **kwargs,
        )
    
    if data_type == "xor":
        return generate_xor_data(
            n_samples = n_samples,
            noise_propability = noise_propability,
            random_state = random_state,
            **kwargs,
        )
    
    if data_type == "circle":
        return generate_circle_data(
            n_samples = n_samples,
            noise_propability = noise_propability,
            random_state = random_state,
            **kwargs,
        )
    
    raise ValueError("data_type должен быть одним из: 'linear', 'xor', 'circle'")

