import numpy as np
from matplotlib import pyplot as plt
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
    clf = GDA()
    # *** START CODE HERE ***

    # Train a GDA classifier
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    #Because of .savefig() in util.plot, we need to change save_path into .png file. 
    #.savefig() from: https://www.geeksforgeeks.org/matplotlib-pyplot-savefig-in-python/
    #.replace() From: https://www.geeksforgeeks.org/python-string-replace/ 
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept = False)
    #x_eval[:,1] = np.exp(-np.power(x_eval[:,1],2))*5
    #x_eval[:,1] = np.log(np.abs(x_eval[:,1]))
    #x_eval[:,1] = np.abs(x_eval[:,1])**0.5 
   
    
    data_plot = save_path.replace('txt', 'png')
    util.plot(x_eval, y_eval, clf.theta, data_plot)

    # Use np.savetxt to save outputs from validation set to save_path
    prediction = clf.predict(x_eval)
    #counter = 0
    #for i in range(np.shape(y_eval)[0]):
      #  if(np.abs(prediction[i] - y_eval[i]) <= 0.5):
     #       counter = counter + 1
    #accuracy = counter/np.shape(y_eval)[0]
    #print(accuracy)
    np.savetxt(save_path, prediction)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True, sigma = None, mu_0 = None, mu_1 = None):
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
        self.sigma = sigma
        self.mu_0 = mu_0
        self.mu_1 = mu_1
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        #Use the derived four parameters of maximum likelkihood function 
       
        N = x.shape[0]
        n = x.shape[1]
        Y = y
        self.theta = np.zeros(n+1)

        phi = 1/N* np.sum(y)
       # y = y.reshape(-1,1) 
       # mu is 2x1 vector so that we need to make the mu into 2x1 vector by x.T @ (y == 0 or 1)
        mu_0 = x.T@(y==0)/ np.sum(y == 0)
        mu_1 = x.T@(y == 1)/ np.sum(y == 1)
       
        #mu_yi --> 800x2 matrix in order to operate (x-mu_yi).T and (x-mu_yi)        
        #np.vstack from: https://stackoverflow.com/questions/68102569/numpy-create-mx2-matrix-based-on-mx1-matrix
        #y.astype(int) from: https://stackoverflow.com/questions/62329081/python-indexerror-arrays-used-as-indices-must-be-of-integer-or-boolean-type
        mu_yi = np.vstack((mu_0,mu_1))[y.astype(int),:]
       
       #@ from: https://blog.finxter.com/numpy-matmul-operator/
       #Sigma matrix -> 2x2 matrix
       #(x-mu_yi) -> 800x2 matrix
       #It should be (x-mu_yi).T@(x-mu_yi) to make sigma 2 x 2 matrix
        sigma = (x-mu_yi).T@(x-mu_yi)* 1/N
        sigma_inv = np.linalg.inv(sigma)

        self.sigma = sigma
        self.mu_0 = mu_0
        self.mu_1 = mu_1

        #Update Theta from the derived four parameters
        a = (0.5*(mu_0.T@sigma_inv@mu_0 - mu_1.T@sigma_inv@mu_1) -np.log(1-phi)+np.log(phi))
        b = sigma_inv@(mu_1-mu_0)
        self.theta[0] = a
        self.theta[1:] = b[0:]

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        #put the updated function into the sigmoid function.
        # prob = 1/exp(-theta.Tx+theta_0) 
       
        prob = 1/(1 + np.exp(-(x.dot(self.theta[1:])+self.theta[0])))

        return prob

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
