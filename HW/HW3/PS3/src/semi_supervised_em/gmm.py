import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

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
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n = np.shape(x)[0]
    group = np.random.choice(K, n)
    mu = [np.mean(x[group == g, :], axis=0) for g in range(K)]
    sigma = [np.cov(x[group == g, :].T) for g in range(K)]

    
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full((K,), fill_value=(1 / K))


    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n, K), fill_value=(1 / K))

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    n = np.shape(x)[0]
    d = np.shape(x)[1]
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w

        for i in range(n):
            D = 0
            for j in range(K):
                constant = 1 / ((2. * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma[j])))
                N = constant * np.exp(-0.5 * (x[i,:]-mu[j]).T@(np.linalg.inv(sigma[j])@(x[i,:]-mu[j])))*phi[j]
                w[i,j] = N
            w[i,:] = w[i,:]/np.sum(w[i,:])

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):

            phi[j] = np.sum(w[:,j])/n

            mu_j = np.zeros(d)
            shape = (d,d)
            sigma_j = np.zeros(shape)

            #collect mu and sigma values
            for i in range(n):
                mu_j += w[i,j]*x[i,:]
                sigma_j += w[i,j] * np.outer((x[i,:]-mu[j]),(x[i,:]-mu[j]))

            mu[j] = mu_j / np.sum(w[:,j])

            #Sigma = sum_{i=1}^{n} W_j^{i}(x^i-mu_j)(x^i-mu_j).T / sum_i=1^n W_j^i
            sigma[j] = sigma_j / np.sum(w[:,j])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll

        log_like = 0

        for i in range(n):
            temp = 0
            for j in range(K):
                constant = 1 / ((2. * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma[j])))
                temp += constant * np.exp(-0.5 * (x[i,:]-mu[j]).T@(np.linalg.inv(sigma[j])@(x[i,:]-mu[j])))*phi[j]
            log_like += np.log(temp)
        
        ll = log_like

        it += 1
        print("Iteration: {}, {}".format(it, ll))
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    n = np.shape(x)[0]
    d = np.shape(x)[1]
    n_tilde = np.shape(x_tilde)[0]
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        for i in range(n):
        
            for j in range(K):
                constant = 1 / ((2. * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma[j])))
                N = constant * np.exp(-0.5 * (x[i,:]-mu[j]).T@(np.linalg.inv(sigma[j])@(x[i,:]-mu[j])))*phi[j]
                w[i,j] = N
            w[i,:] = w[i,:]/np.sum(w[i,:])

        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        w_sum = np.sum(w, axis=0)
        sup_sum = [np.sum(z_tilde == j) for j in range(K)]
        for j in range(K):
            phi[j] = (w_sum[j] + alpha * sup_sum[j]) / (n + alpha * n_tilde)

            w_j = w[:, j:j + 1]
            mu[j] = ((np.sum(w_j * x, axis=0)
                     + alpha * np.sum(x_tilde[(z_tilde == j).squeeze(), :], axis=0))
                 / (np.sum(w_j) + alpha * sup_sum[j]))

            sigma[j] = np.zeros_like(sigma[j])
            for i in range(n):
                a = x[i] - mu[j]
                sigma[j] += w[i, j] * np.outer(a,a)

            for k in range(n_tilde):
                if z_tilde[k] == j:
                    a = x_tilde[k] - mu[j]
                    sigma[j] += alpha * np.outer(a,a)
            sigma[j] /= (np.sum(w_j) + alpha * sup_sum[j])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll

        unsup_log_like = 0
        sup_log_like = 0
        temp_sup = 0
        for i in range(n):
            temp = 0
            for j in range(K):
                constant = 1 / ((2. * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma[j])))
                inv_sigma = np.linalg.inv(sigma[j])
                a = (x[i]-mu[j])
                temp += constant * np.exp(-0.5 * a.T@(inv_sigma)@ a)*phi[j]
            unsup_log_like += np.log(temp)

        for i in range(n_tilde):
            z = int(z_tilde[i])
            constant = 1 / ((2. * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(sigma[z])))
            temp_sup += constant * np.exp(-0.5 * (x_tilde[i]-mu[z]).T@(np.linalg.inv(sigma[z])@(x_tilde[i]-mu[z])))*phi[z]
            sup_log_like += np.log(temp)
        
        
        ll = unsup_log_like + alpha*sup_log_like

        it += 1
        print("Iteration: {}, {}".format(it, ll))

        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


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
        #main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
