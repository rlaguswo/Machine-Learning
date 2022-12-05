import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
      
        self.theta = np.linalg.solve(X.T @ X, X.T.dot(y)) #XTX_inv * X.T.dot(y)
        
       

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        # np.hstack from: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=wideeyed&logNo=221205882962
        x = X[:,1]
        x = x.reshape(-1,1)
        x_0 = np.power(x,0)
        phi = np.hstack((x_0,x))
        for i in range(2, k+1):
            a = np.power(x,i)
            phi = np.hstack((phi,a))
        
        return phi
     
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        x = X[:,1]
        x = x.reshape(-1,1)
        phi = self.create_poly(k,X)
        sine = np.sin(X)
        phi_sine = np.hstack((phi,sine))

        return phi_sine
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        prediction = X.dot(self.theta)
        return prediction
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        clf = LinearModel()

        #if the argument is sine
        if (sine == True):
           train_plot = clf.create_sin(k, train_x)
           data_plot = clf.create_sin(k, plot_x)

        #if the argument is not sine  
        else:
            train_plot = clf.create_poly(k, train_x)
            data_plot = clf.create_poly(k, plot_x)
      
        clf.fit(train_plot, train_y)
        prediction = clf.predict(data_plot)
        
        
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plot_y = prediction
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all experiments
    '''
    # *** START CODE HERE ***
    #part (b)
    run_exp(train_path, sine=False, ks=[3], filename='plot_b.png')
    #part(c)
    run_exp(train_path, sine=False, ks=[3, 5, 10, 20], filename='plot_c.png')
    #part(d)
    run_exp(train_path, sine=True, ks=[0,1, 2, 3, 5, 10, 20], filename='plot_d.png')
    #part (e)
    run_exp(small_path, sine= False, ks=[1,2,5,10,20], filename='plot_e.png')

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
