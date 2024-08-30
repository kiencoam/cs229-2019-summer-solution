import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    z_init = np.random.randint(0, K, size=x.shape[0])
    mu = np.zeros((K, x_all.shape[1]))
    sigma = np.zeros((K, x_all.shape[1], x_all.shape[1]))
    for j in range(K):
        x_tilde_j = x_tilde[np.where(z_tilde == j)[0],:]
        x_unlabeled_j = x[np.where(z_init == j)[0],:]
        x_j = np.concatenate((x_tilde_j, x_unlabeled_j), axis=0)
        mu[j] = np.mean(x_j, axis=0)
        sigma[j] = np.mean((x_j.reshape((-1, x_j.shape[1], 1)) - mu[j].reshape((1, -1, 1)))
                           * (x_j.reshape((-1, 1, x_j.shape[1])) - mu[j].reshape((1, 1, -1))), axis=0)
        sigma[j] += np.eye(sigma[j].shape[0]) * 1e-6
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    z = np.concatenate((z_init, z_tilde.flatten()))
    phi = np.zeros(K)
    for j in range(K):
        phi[j] = np.where(z == j)[0].shape[0] / z.shape[0]
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.zeros((x.shape[0], K))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    n = x.shape[0]
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n, d).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, of shape (k, d).
        sigma: Initial cluster covariances, of shape (k, d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    mat = probabilities_of_x_z(x, phi, mu, sigma)

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        it += 1

        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        mat_sum = np.sum(mat, axis=1).reshape((-1, 1))
        mat_sum[mat_sum == 0] = 1e-10
        w = mat / mat_sum

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.mean(w, axis=0)

        mu = np.matmul(w.T, x) / np.sum(w, axis=0).reshape((-1, 1))

        for j in range(K):
            sigma[j] = np.zeros(sigma[j].shape)
            for i in range(x.shape[0]):
                sigma[j] += w[i, j] * np.outer(x[i] - mu[j], x[i] - mu[j])
            sigma[j] /= np.sum(w[:, j])
            sigma[j] += np.eye(sigma[j].shape[0]) * 1e-6

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll

        mat = probabilities_of_x_z(x, phi, mu, sigma)
        ll = np.sum(np.log(np.sum(mat, axis=1) + 1e-10))
        print("Iteration {}: ll = {}".format(it, ll))
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n, d).
        x_tilde: Design matrix of labeled examples of shape (n_tilde, d).
        z_tilde: Array of labels of shape (n_tilde, 1).
        w: Initial weight matrix of shape (n, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, of shape (k, d).
        sigma: Initial cluster covariances, of shape (k, d, d).

    Returns:
        Updated weight matrix of shape (n, d) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    mat = probabilities_of_x_z(x, phi, mu, sigma)

    # Number of examples respect to observed j-th cluster
    count = np.zeros(phi.shape[0])
    for j in range(K):
        count[j] = np.where(z_tilde == j)[0].shape[0]

    # Total of vectors in each observed j-th cluster
    sum_over_cluster = np.zeros((phi.shape[0], x.shape[1]))
    for j in range(K):
        sum_over_cluster[j] = np.sum(x_tilde[np.where(z_tilde == j)[0]], axis=0)

    # Total of covariance matrices in each observed j-th cluster
    covariance_sum_over_cluster = np.zeros((phi.shape[0], x.shape[1], x.shape[1]))
    for j in range(K):
        covariance_sum_over_cluster[j] = np.sum(
            (x_tilde[np.where(z_tilde == j)[0]].reshape((-1, x.shape[1], 1)) - mu[j].reshape((1, -1, 1))) * (
                        x_tilde[np.where(z_tilde == j)[0]].reshape((-1, 1, x.shape[1])) - mu[j].reshape((1, 1, -1))),
            axis=0)

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        it += 1

        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        mat_sum = np.sum(mat, axis=1).reshape((-1, 1))
        mat_sum[mat_sum == 0] = 1e-10
        w = mat / mat_sum

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = (np.sum(w, axis=0) + alpha * count) / (x.shape[0] + alpha * x_tilde.shape[0])

        mu = (w.T @ x + alpha * sum_over_cluster) / (np.sum(w, axis=0) + alpha * count).reshape((-1, 1))

        for j in range(K):
            sigma[j] = np.zeros(sigma[j].shape)
            for i in range(x.shape[0]):
                sigma[j] += w[i, j] * np.outer(x[i] - mu[j], x[i] - mu[j])
            sigma[j] = ((sigma[j] + alpha * covariance_sum_over_cluster[j])
                        / (np.sum(w[:, j]) + alpha * count[j]))
            sigma[j] += np.eye(sigma[j].shape[0]) * 1e-6

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll

        mat = probabilities_of_x_z(x, phi, mu, sigma)

        # Calculate log_likelihood tilde
        ll_tilde = 0
        mat_tilde = probabilities_of_x_z(x_tilde, phi, mu, sigma)
        for i in range(x_tilde.shape[0]):
            cluster = z_tilde.flatten()[i].astype(int)
            ll_tilde += np.log(mat_tilde[i, cluster] + 1e-10)

        ll = np.sum(np.log(np.sum(mat, axis=1) + 1e-10)) + alpha * ll_tilde
        print("Iteration {}: ll = {}".format(it, ll))
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def probabilities_of_x_z(x, phi, mu, sigma):
    """Calculate probabilities P(x(i), z(i)=j | phi, mu, sigma).

    Args:
        x: Design matrix of unlabeled examples of shape (n, d).
        phi: Probability of a example belonging to the j-th Gaussian in the mixture, of shape (k,).
        mu: Cluster means, of shape (k, d)
        sigma: Cluster covariances, of shape (k, d, d)

    Returns:
        A matrix which each element is P(x(i), z(i)=j | phi, mu, sigma), of shape (n, k).
    """
    mat = np.zeros((x.shape[0], phi.shape[0]))
    for j in range(K):
        try:
            mat[:, j] = (phi[j] * np.exp(
                -.5 * np.diag((x - mu[j].reshape((1, -1))) @ np.linalg.inv(sigma[j]) @ (x - mu[j].reshape((1, -1))).T))
                         / (np.sqrt(np.linalg.det(sigma[j])) + 1e-10))
        except np.linalg.LinAlgError:
            print(f"Covariance matrix at index {j} is singular.")
            mat[:, j] = 1e-10
    return mat

def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=True, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # *** END CODE HERE ***
