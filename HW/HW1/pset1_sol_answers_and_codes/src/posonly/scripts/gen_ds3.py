import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

C = 0.2  # P(y = 1 | t = 1)
DATA_DIR = './'
np.random.seed(229)


def generate_gaussian(num_examples=500):
    """Generate dataset where each class is sampled from a bivariate Gaussian."""

    # Set parameters for each class
    class_labels = (0., 1.)
    mus = ([-2, -2], [1.5, 2])
    sigmas = ([[3.0, 0.], [0., 3.0]], [[0.2, 0.4], [0.4, 2.5]])

    # Generate dataset
    examples = []
    for class_label, mu, sigma in zip(class_labels, mus, sigmas):
        # Sample class from Gaussian
        class_examples = np.random.multivariate_normal(mu, sigma, num_examples // len(class_labels))

        # Add each example to the list
        for x in class_examples:
            x_dict = {f'x_{i+1}': x_i for i, x_i in enumerate(x)}
            x_dict['t'] = class_label
            x_dict['y'] = 1 if class_label == 1 and np.random.uniform() < C else 0
            examples.append(x_dict)

    df = pd.DataFrame(examples)

    return df


def plot_dataset(df, output_path):
    """Plot a 2D dataset and write to output_path."""
    xs = np.array([[row['x_1'], row['x_2']] for _, row in df.iterrows()])
    ys = np.array([row['y'] for _, row in df.iterrows()])

    plt.figure(figsize=(12, 8))
    for x_1, x_2, y in zip(xs[:, 0], xs[:, 1], ys):
        marker = 'x' if y > 0 else 'o'
        color = 'red' if y > 0 else 'blue'
        plt.scatter(x_1, x_2, marker=marker, c=color, alpha=.5)
    plt.savefig(output_path)


if __name__ == '__main__':

    for split, n in [('train', 1250), ('valid', 125), ('test', 126)]:
        gaussian_df = generate_gaussian(num_examples=n)
        gaussian_df.to_csv(os.path.join(DATA_DIR, f'{split}.csv'), index=False)
        if split == 'train':
            plot_dataset(gaussian_df, os.path.join(DATA_DIR, 'plot.eps'))
