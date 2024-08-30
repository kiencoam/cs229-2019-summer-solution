import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression(verbose=False)
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, save_path)

    # Use np.savetxt to save predictions on eval set to save_path
    y_pred = clf.predict(x_valid)
    right_case = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_valid[i]:
            right_case += 1
    right_case /= y_pred.shape[0]
    print("Predicted probability: ", right_case)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for count in range(self.max_iter):
            # Calculate sigma for current theta
            z = np.matmul(x, self.theta)
            sigma = np.zeros(z.size)
            for i in range(z.size):
                sigma[i] = 1 / (1 + np.exp(-z[i]))

            # Print loss value
            if self.verbose:
                loss_function = 0
                for i in range(x.shape[0]):
                    if y[i] == 1:
                        loss_function += 1 - sigma[i]
                    else:
                        loss_function += sigma[i]
                loss_function /= x.shape[0]
                print("Iteration", count, "Loss: ", loss_function)

            # Calculate derivative of J for current theta
            derivative = np.matmul(x.T, sigma - y)

            # Calculate Hessian matrix for current theta
            hessian = np.zeros((x.shape[1], x.shape[1]))
            for i in range(x.shape[0]):
                x_i = np.expand_dims(x[i][:], axis=1)
                hessian += sigma[i] * np.matmul(x_i, x_i.T)

            # Calculate new theta
            new_theta = self.theta - np.matmul(np.linalg.inv(hessian), derivative)

            # Check whether current theta is optimization solution or not
            if np.linalg.norm(self.theta - new_theta) < self.eps:
                break
            else:
                self.theta = new_theta
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
                y[i] = 1.0
            else:
                y[i] = 0.0
        return y
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_1.png')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_2.png')
