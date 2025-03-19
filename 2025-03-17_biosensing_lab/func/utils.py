from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray 

def generate_data(n_sample, n_features, centers, cluster_std, noise_fraction):
    """
    Generates synthetic data with weighted samples in each cluster.

    Parameters:
    - n_sample: Total number of samples
    - n_features: Number of features for each data point
    - centers: List of cluster centers
    - cluster_std: Standard deviation of the clusters
    - noise_fraction: Fraction of noise to be added
    - cluster_weights: List of weights for each cluster, should sum to 1
    """
    cluster_weights = [0.4, 0.2, 0.4]
    center_box =[(-5,10), (-15,15), (-30,30)]
    # Ensure that the weights sum to 1
    if sum(cluster_weights) != 1:
        raise ValueError("The cluster weights must sum to 1.")
    
    # Calculate the number of samples for each cluster based on the weights
    n_samples_per_cluster = [int(weight * n_sample) for weight in cluster_weights]

    # Generate data for each cluster with the specified number of samples
    X_all = []
    for i, n_samples in enumerate(n_samples_per_cluster):
        X_cluster, _ = make_blobs(n_samples=n_samples, cluster_std=cluster_std[i], center_box=center_box[i],
                                  centers=[centers[i]], n_features=n_features, random_state=6)
        X_all.append(X_cluster)
    
    # Concatenate the data from all clusters
    X = np.vstack(X_all)
    
    # Add noise
    n_noise = int(n_sample * noise_fraction)
    noise_data = np.random.uniform(low=-10, high=150, size=(n_noise, n_features))
    X_noisy = np.concatenate((X, noise_data))
    
    return X_noisy


def plot_image(ax, img ,title= "", is_gray=False):
    """
    Plots an image using matplotlib.

    Parameters:
    - ax: Matplotlib axis object to plot on
    - img: Image to be plotted
    - title: Title for the image plot (optional)
    - is_gray: Whether the image should be plotted in grayscale (optional, default=False)
    """
    if is_gray:
        ax.imshow(img, cmap= plt.cm.gray)
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

def rgb_to_gray(img):
    """
    Converts an RGB image to grayscale using skimage.

    Parameters:
    - img: RGB image"""

    img= img[:, :, :3] # make sure that there is not alpha channel in the image
    return rgb2gray(img)

def plot_histogram(ax,gray_img, title=""):
    """
    Plot a histogram of grayscale values for a given grayscale image.#+

    Parameters:
    ax: Matplotlib axis object to plot on
    gray_img (numpy.ndarray): A 2D numpy array representing the grayscale image.#+
    title (str, optional): The title for the histogram plot. Defaults to an empty string.#+

    Returns:
    None: This function does not return any value. It displays the histogram plot.#+
    """

    ax.hist(gray_img.flatten(), bins=256, range=(0, 1))
    ax.set_title(f'{title}')
    ax.set_xlabel('Grayscale Value')
    ax.set_ylabel('Pixels')
  

def gmm_clustering(df):
    n_clusters = 3  # Set number of clusters
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df['Cluster'] = gmm.fit_predict(df[['avg_r', 'avg_g']])

    # Check the cluster assignment
    print("Cluster Counts:")
    print(df['Cluster'].value_counts())

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(df['avg_r'], df['avg_g'], c=df['Cluster'], cmap='viridis', marker='o', s= 0.5)
    plt.xlabel('avg_r')
    plt.ylabel('avg_g')
    plt.title('GMM Clustering Results')
    plt.show()