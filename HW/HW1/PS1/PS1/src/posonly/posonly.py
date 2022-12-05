import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

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
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # Specific Intructions:
    # write a logistic regression classifer that uses x1 and x2 as input features, and train it using the t-labels.
    # Output the trained model's predictions on the test set to the file specified in the code
    # Create a plot to visualize the test set with x1 on the horizontal axis and x2 on the vertical axis
    clf_a = LogisticRegression()
    x_train_a, y_train_a = util.load_dataset(train_path, label_col='t', add_intercept= True)
    clf_a.fit(x_train_a, y_train_a)
    x_eval_a, y_eval_a = util.load_dataset(test_path, label_col='t', add_intercept= True)


    data_plot_a = output_path_true.replace('txt', 'png')
    util.plot(x_eval_a, y_eval_a, clf_a.theta, data_plot_a)

    predictions_a = clf_a.predict(x_eval_a)
    np.savetxt(output_path_true, predictions_a)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Specific Instructions:
    # you only have access to the y-labels at training time
    # Output the predictions on the test set to the appropriate file (as described in the code comments).
    # Create a plot to visualize the test set with x1 on the horizontal axis and x2 on the vertical axis.
    clf_b = LogisticRegression()

    x_train_b, y_train_b = util.load_dataset(train_path, label_col = 'y', add_intercept= True)
    clf_b.fit(x_train_b, y_train_b)
    
    x_eval_b, y_eval_b = util.load_dataset(test_path, label_col = 't', add_intercept= True)
    data_plot_b = output_path_naive.replace('txt','png')
    util.plot(x_eval_b ,y_eval_b, clf_b.theta, data_plot_b)

    predictions_b = clf_b.predict(x_eval_b)
    np.savetxt(output_path_naive, predictions_b)

    # Part (f): Apply correction factor using validation set and test on true labels
    clf_f = LogisticRegression()
    x_eval_f, y_eval_f = util.load_dataset(valid_path, label_col ='y', add_intercept= True)
    clf_f.fit(x_train_b, y_train_b)
    V_plus =len(x_eval_f[y_eval_f == 1])
    predictions_f = clf_f.predict(x_eval_f[y_eval_f == 1])
    
    alpha = 1/V_plus * np.sum(predictions_f)
    predict = predictions_b/alpha

    
    data_plot_f = output_path_adjusted.replace('txt','png')
    util.plot(x_eval_a, y_eval_a, clf_f.theta, data_plot_f, correction= alpha)
    np.savetxt(output_path_adjusted, predict)

    
    # Plot and use np.savetxt to save outputs to output_path_adjusted

    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
