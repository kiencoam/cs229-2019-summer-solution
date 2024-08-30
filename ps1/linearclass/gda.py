import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, save_path)

    # Use np.savetxt to save outputs from validation set to save_path
    y_pred = clf.predict(x_valid)
    predicted_probability = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_valid[i]:
            predicted_probability += 1
    probability = predicted_probability / y_pred.shape[0]
    print("Predicted probability: ", probability)
    # *** END CODE HERE ***

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi
        positive_case = 0
        for i in range(y.shape[0]):
            if y[i] == 1:
                positive_case += 1
        phi = positive_case / x.shape[0]

        # Find mu_1
        mu_1 = np.zeros(x.shape[1])
        for i in range(x.shape[0]):
            if y[i] == 1:
                mu_1 += x[i]
        mu_1 /= positive_case

        # Find mu_0
        mu_0 = np.zeros(x.shape[1])
        for i in range(x.shape[0]):
            if y[i] == 0:
                mu_0 += x[i]
        mu_0 /= 1 - positive_case

        # Find sigma
        sigma = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            if y[i] == 1:
                z = np.expand_dims(x[i] - mu_1, axis=1)
                sigma += np.matmul(z, z.T)
            else:
                z = np.expand_dims(x[i] - mu_0, axis=1)
                sigma += np.matmul(z, z.T)
        sigma /= x.shape[0]

        # Write theta in terms of the parameters
        self.theta = np.zeros(x.shape[1] + 1)
        self.theta[0] = - np.log(1/phi - 1)
        self.theta[0] += np.matmul(mu_0, np.matmul(np.linalg.inv(sigma), mu_0)) / 2
        self.theta[0] -= np.matmul(mu_1, np.matmul(np.linalg.inv(sigma), mu_1)) / 2
        self.theta[1:] = np.matmul(np.linalg.inv(sigma).T, mu_1 - mu_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if np.dot(x[i], self.theta) >= 0:
                y[i] = 1
            else:
                y[i] = 0
        return y
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.png')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.png')
