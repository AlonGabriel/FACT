import pathlib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from train import make_model
import joblib
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



class SimpleLogger:
    @staticmethod
    def info(message):
        print(f"[INFO] {message}")

    @staticmethod
    def warning(message):
        print(f"[WARNING] {message}")


class DESIModelDeployment:
    def __init__(self, base_model, projection_head, random_init, prediction_head, checkpoint):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = checkpoint
        self.model = self.make_model(base_model, projection_head, prediction_head, random_init, checkpoint)

        
    def make_model(self, base_model, projection_head, prediction_head, random_init, checkpoint, supervised=False):
        model = make_model(base_model=base_model, projection_head=projection_head, random_init=random_init, prediction_head=prediction_head, 
                           checkpoint=checkpoint, _log=SimpleLogger, supervised=supervised)
        model = model.to(self.device)
        if model.pred_head is None:
            SimpleLogger.warning("Prediction head is None. Check model initialization.")
        model.eval()
        return model
    
    
    def predict(self, aligned_peaks_norm, batch_size=150):
        """
        Have model make predictions for each peak.
        Inputs:
            aligned_peaks_norm - aligned and normalized peak array to the reference m/z list
            batch_size - batch size (int) to be fed into the model
        Output:
           predictions - array of predictions for each peak
        """
        predictions = []
        embeddings = []
        total_batches = int(np.ceil(aligned_peaks_norm.shape[0] / batch_size))

        # Add a progress bar for prediction
        for i in tqdm(range(0, aligned_peaks_norm.shape[0], batch_size), total=total_batches, desc="Predicting"):
            batch = aligned_peaks_norm.iloc[i:i + batch_size].to_numpy() if isinstance(aligned_peaks_norm, pd.DataFrame) else aligned_peaks_norm[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                model_output = self.model(batch_tensor)
                batch_predictions = model_output.predictions.cpu().numpy()
                batch_embeddings = model_output.embeddings.cpu().numpy()
            # Convert probabilities to class labels
            batch_classes = np.argmax(batch_predictions, axis=1)
            predictions.append(batch_classes)
            embeddings.append(batch_embeddings)
        return np.concatenate(predictions, axis=0), np.concatenate(embeddings, axis=0)
    
    
    def reduce_dimensions(self, embeddings, n_components=2):
        """
        Reduce embeddings to 2D using PCA.

        Inputs:
        embeddings: torch.Tensor
            High-dimensional embeddings.
        n_components: int
            Number of dimensions to reduce to (default is 2).

        Outputs:
        reduced_embeddings: np.ndarray
            Reduced embeddings (2D or 3D).
        """
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings
    
    
    def reduce_dimensions_umap(self, embeddings, n_components=2, random_state=42):
        """
        Reduce embeddings to 2D using UMAP.

        Inputs:
            embeddings - High-dimensional embeddings (NumPy array or Tensor).
            n_components - Number of dimensions to reduce to (default is 2).
            random_state - Random seed for reproducibility.
        Outputs:
            reduced_embeddings - UMAP-reduced embeddings (2D or 3D as NumPy array).
        """
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings
    
    
    def plot_scatter(self, embeddings, labels=None, title="Embedding Scatter Plot", save_path=None):
        """
        Scatter plot of embeddings.
        Inputs:
            embeddings - 2D embeddings (NumPy array)
            labels - optional labels for coloring the points
            title - title of the plot
            save_path - path to save the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Label" if labels is not None else "Intensity")
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        
    def plot_scatter_3d(self, embeddings, labels=None, title="3D Embedding Scatter Plot", save_path=None):
        """
        3D scatter plot of embeddings.
        Inputs:
            embeddings - 3D embeddings (NumPy array).
            labels - Optional labels for coloring the points.
            title - Title of the plot.
            save_path - Path to save the plot (optional).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
            c=labels, cmap='viridis', alpha=0.7
        )
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")

        # Add a colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label("Label" if labels is not None else "Intensity")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()    
        
        
    def plot_contour(self, embeddings, title="Embedding Contour Plot", save_path=None):
        """
        Contour plot of embeddings.
        Inputs:
            embeddings - 2D embeddings (NumPy array)
            title - title of the plot
            save_path - path to save the plot (optional)
        """
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        plt.figure(figsize=(8, 6))
        plt.tricontourf(x, y, z, levels=20, cmap="viridis")
        plt.colorbar(label="Density")
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        
    def visualize_embeddings(self, embeddings, dim_y, dim_x, save_path, dimension=None, metric='mean', mask_path=None):
        """
        Create a plot of pixel-level embeddings from the model.

        Inputs:
            embeddings - array of embeddings for each pixel (NumPy array of shape [num_pixels, num_dimensions]).
            dim_y - Y dimension of the original MSI (int).
            dim_x - X dimension of the original MSI (int).
            save_path - Location to save the plot.
            dimension - Specific embedding dimension to visualize (int). If None, computes a summary metric.
            metric - Summary metric for the embeddings if `dimension` is None. Options: 'L2', 'mean', 'max'.
        """
        print("Are there NaN values in embeddings?:", np.isnan(embeddings).any())
        print("Are there infinite values in embeddings?:", np.isinf(embeddings).any())
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        # Reshape embeddings into the original image shape
        if dimension is not None:
            # Visualize a specific embedding dimension
            emb_img = embeddings[:, dimension].reshape((dim_y, dim_x), order='C')
        else:
            # Compute a summary metric for embeddings
            if metric == 'L2':
                emb_img = np.linalg.norm(embeddings, axis=1).reshape((dim_y, dim_x), order='C')
            elif metric == 'mean':
                emb_img = embeddings.mean(axis=1).reshape((dim_y, dim_x), order='C')
            elif metric == 'max':
                emb_img = embeddings.max(axis=1).reshape((dim_y, dim_x), order='C')
            else:
                raise ValueError("Unsupported metric. Choose from 'L2', 'mean', or 'max'.")
            
        # Normalize the embedding image for better visualization
        if np.max(emb_img) - np.min(emb_img) > 1e-6:
            emb_img = (emb_img - np.min(emb_img)) / (np.max(emb_img) - np.min(emb_img))
        else:
            print("Skipping normalization due to small range in embedding values.")
            emb_img = emb_img - np.min(emb_img)  # Center values only

        # Plot the embedding image
        fig, ax = plt.subplots()
        cax = ax.imshow(emb_img, cmap='viridis', interpolation='nearest')
        ax.axis('off')

        # Add a colorbar
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(cax, cax=cbar_ax)
        cbar.set_label("Embedding Value" if dimension is not None else f"{metric} Metric")

        plt.title(f"Embedding Visualization ({'Dimension ' + str(dimension) if dimension is not None else metric})")
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()


        
if __name__ == "__main__":
    # Load your dataset
    data = np.load("Datasets/aligned_labeled.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Initialize the deployment class
    desi_model = DESIModelDeployment(
        base_model="laion/clap-htsat-unfused",
        projection_head=True,
        random_init=False,
        prediction_head=[128, 2],
        checkpoint="sample_norm_SimCLR/checkpoint_17_loss=-4.5714.pt"
    )

    # Predict and extract embeddings
    _, embeddings = desi_model.predict(X_train)

    # Reduce dimensions
    reduced_embeddings = desi_model.reduce_dimensions(embeddings, n_components=2)
    #reduced_embeddings_umap = desi_model.reduce_dimensions_umap(embeddings, n_components=3)
    
    # Normalize and Cluster
    normalized_embeddings = StandardScaler().fit_transform(reduced_embeddings)
    n_clusters = 2  # Adjust based on the number of expected segments
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    # Visualize Clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='tab10', s=5)
    plt.colorbar(label="Cluster")
    plt.title("Clustered Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("Plots/clusters_scatter", bbox_inches='tight', dpi=600)
    
    sil_score = silhouette_score(normalized_embeddings, cluster_labels)
    print(f"Silhouette Score: {sil_score}")

    # Visualize embeddings
    #desi_model.plot_scatter(reduced_embeddings, labels=y_train, title="2D Scatter Plot of Embeddings", save_path="Plots/SimCLR_snPCA_scatterplot")

#     desi_model.plot_scatter(reduced_embeddings_umap, labels=y_test, title="Scatter Plot of Embeddings", save_path="Plots/umap3D_scatterplot")
#     desi_model.plot_contour(reduced_embeddings_umap, title="Contour Plot of Embeddings", save_path="Plots/umap3D_contourplot")