#@ from: https://blog.finxter.com/numpy-matmul-operator/
import numpy as np
from matplotlib import pyplot as plt
import util 



def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept= True)
    # *** START CODE HERE ***
    clf = LogisticRegression()  
    # Train a logistic regression classifier
    clf.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept = True)

    #Because of .savefig() in util.plot, we need to change save_path into .png file. 
    #.savefig() from: https://www.geeksforgeeks.org/matplotlib-pyplot-savefig-in-python/
    #Source From: https://www.geeksforgeeks.org/python-string-replace/ 
    data_plot = save_path.replace('txt', 'png')
    util.plot(x_eval, y_eval, clf.theta, data_plot)

    # Use np.savetxt to save predictions on eval set to save_path
    predictions = clf.predict(x_eval)
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
    
        n = np.shape(x)[1]
        N = np.shape(x)[0]
        #Initialize theta with R^n zero vect
        self.theta = np.zeros(n)

        for i in range(self.max_iter):
            theta_kth = np.array(self.theta)
           # print(x.shape)

            #gradient: -1/n*(y-sigmoid(x.dot(theta_kth))*x_j
            gradient = 1/N* x.T.dot(((1/(1 + np.exp(-x.dot(self.theta)))))-y)
            #print(x.T.shape)

            #Hessian:  1/n*(sigmodi(x.dot(theta_kth)))*(1-sigmoid(x.dot(self.theta)))*x*x.T
            a = (1/(1 + np.exp(-x.dot(self.theta))))*(1-1/(1 + np.exp(-x.dot(self.theta))))

           # print(a.shape)
           # print(a)
           # print(x.T)
            #b = a * x.T
            #print(b.shape)
            #print(b)
            #print(x.T.shape)
            #print((a*x.T).shape)
            
            Hessian = 1/N *  np.dot(a*x.T,x)
           
            #update Theta
            Hessian_inv = np.linalg.inv(Hessian)
            self.theta = theta_kth - Hessian_inv.dot(gradient)
           
            #Source From: https://stackoverflow.com/questions/886633/get-the-1-norm-of-a-vector-in-python
            dst = np.linalg.norm(self.theta - theta_kth, ord = 1)
            if(dst < self.eps):
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        #Pseudo code
        #putting to the sigmoid function for each x in valid_path or vectorize version
        #vectorize entire n

        prob = 1/(1 + np.exp(-x.dot(self.theta)))

        return prob

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
