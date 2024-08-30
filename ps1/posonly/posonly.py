import numpy as np
import util
import sys

sys.path.append('../linearclass')

# NOTE : You need to complete p01b_logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'

def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    clf_true = LogisticRegression()
    clf_true.fit(x_train, t_train)

    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    util.plot(x_valid, t_valid, clf_true.theta, output_path_true)

    # Part (b): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    clf_naive = LogisticRegression()
    clf_naive.fit(x_train, y_train)

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    util.plot(x_valid, t_valid, clf_naive.theta, output_path_naive)

    # Part (f): Apply correction factor using validation set and test on true labels
    alpha = 0
    y_train_observed = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 1:
            y_train_observed += 1
            alpha += 1 / (1 + np.exp(-np.dot(x_train[i], clf_naive.theta)))
    alpha /= y_train_observed

    # Plot and use np.savetxt to save outputs to output_path_adjusted
    util.plot(x_valid, t_valid, clf_naive.theta, output_path_adjusted, correction=alpha)

    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.png')
