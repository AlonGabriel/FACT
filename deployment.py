"""
deployment.py

This script performs full model deployment for DESI-MSI data, including:
- Loading and aligning raw spectral data (.txt or .csv)
- Normalizing peak intensities
- Predicting pixel-level class labels using a pretrained model
- Generating segmentation visualizations
- Optionally visualizing embeddings and clustering spatial features

Designed for end-to-end inference on MSI tissue slides using trained classification models.

Typical usage:
    python deployment.py --desi_data_path <path> --save_path <output> --checkpoint <model.pt> --mz_ref <ref.npy>

Author:
    Alon Gabriel, 2025
"""

import argparse
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
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SimpleLogger:
    @staticmethod
    def info(message):
        print(f"[INFO] {message}")

    @staticmethod
    def warning(message):
        print(f"[WARNING] {message}")


class DESIModelDeployment:
    def __init__(self, base_model, projection_head, random_init, prediction_head, checkpoint, mz_ref, class_order):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = checkpoint
        self.model = self.make_model(base_model, projection_head, prediction_head, random_init, checkpoint)
        self.mz_ref = np.array(mz_ref, dtype=np.float64)
        self.class_order = class_order
        
        
    def make_model(self, base_model, projection_head, prediction_head, random_init, checkpoint):
        model = make_model(base_model=base_model, projection_head=projection_head, random_init=random_init, prediction_head=prediction_head, 
                           checkpoint=checkpoint, _log=SimpleLogger, supervised=False, reinit_proj_head=True)
        model = model.to(self.device)
        if model.pred_head is None:
            SimpleLogger.warning("Prediction head is None. Check model initialization.")
        model.eval()
        return model
    
    
    def DESI_txt2numpy(self, desi_text):
        data = []
        with open(desi_text, 'r') as read_obj:
                for i,line in enumerate(read_obj):
                        x = line.split()
                        y = [float(num) for num in x]
                        data.append(y)

        ind = np.argsort(data[3]) # data[3] has unsorted m/z values
        mz = np.take_along_axis(np.asarray(data[3]), ind, axis=0) # sort with indices

        x, y = [], []
        peaks = []

        for i in range(4,len(data)-1):
                x.append(data[i][1])
                y.append(data[i][2])
                p = np.asarray(data[i][3:-2])
                p = np.take_along_axis(p, ind, axis=0)
                p = np.expand_dims(p,axis=0)
                peaks.append(p)
        peaks = np.concatenate(peaks,axis=0)
        
        ## find desi data dimension
        t = np.asarray(x)
        t = np.abs(np.diff(t))
        dim_x = int(np.round(np.max(t)/np.min(t)))+1
        t = np.asarray(y)
        dim_y = int(np.round(np.abs(t[0]-t[-1])/np.max(np.abs(np.diff(t)))))+1

        return peaks, mz, dim_y, dim_x

    
    def peak_alignment_to_reference(self, mz_test, peaks_test, mz_ref, thresh=0.05):

        """""
        function:
            MSI m/z alighnemnt
            align m/z values and peaks of an MSI data to a reference m/z list
        inputs:
            mz_test - m/z list of the slide
            peaks_test - peak array of slide
            mz_ref - reference m/z list
            thresh - maximum m/z distance for inclusion
        output:
            aligned_peaks - aligned peak array to the reference m/z list
            final_mz - shows how the slide m/z list is aligned to the reference list 
        author: @moon
        """""
        # calculate peak abundancy
        pmean = peaks_test.mean(axis=0)/peaks_test.mean(axis=0).max()

        # generate list for multiple m/z
        n_mz = len(mz_ref)
        new_mz = []
        for j in range(n_mz):
            new_mz.append([])

        # generate equivalent abundancy list
        new_peaks = []
        for j in range(n_mz):
            new_peaks.append([])

        print('aligning m/z to the reference ...')
        # align current m/z to the reference m/z
        for j in range(len(mz_test)):
            x = mz_test[j]
            y = pmean[j]
            diff = np.abs(mz_ref-x)
            if diff.min()<thresh:
                ind = diff.argmin()
                new_mz[ind].append(x)
                new_peaks[ind].append(y)

        # convert to pandas dataframe for simpler handling
        new_mz_df = pd.DataFrame([new_mz], columns=mz_ref)
        new_peaks_df = pd.DataFrame([new_peaks], columns=mz_ref)

        # eliminate the multiple m/z based on aboundance
        final_mz = np.nan*np.ones(n_mz,)
        for j in range(n_mz):
            mz_cell = new_mz_df[mz_ref[j]][0]
            if len(mz_cell)==0:
                pass
            elif len(mz_cell)==1:
                final_mz[j] = mz_cell[0]
            else:
                pmean_cell = new_peaks_df[mz_ref[j]][0]
                i_abundant = np.array(pmean_cell).argmax()
                final_mz[j] = mz_cell[i_abundant]

        print('aligning the peaks ...')
        # align peaks accordingly
        aligned_peaks = np.nan*np.ones([len(peaks_test),n_mz])
        for j,mz_val in enumerate(tqdm(final_mz)):
            if ~np.isnan(mz_val):
                p_ind = np.where(mz_test == mz_val)[0]
                aligned_peaks[:,j] = peaks_test[:,p_ind].flat

        print('alignment to the m/z reference done!')

        return aligned_peaks, final_mz
    
    
    def tic_normalize(self, peaks):
        tot_ion_cur = np.sum(peaks, axis=1)
        peaks_ticn = np.empty(peaks.shape)
        for i in range(len(peaks)):
            if tot_ion_cur[i]!=0:
                peaks_ticn[i] = peaks[i]/tot_ion_cur[i]
        return peaks_ticn
    
    
    def minmax_normalize_peaks(self, peaks):
        """
        Normalize peak values using MinMax.
        Inputs:
            peaks - peak array of slide
        Output:
            peaks_norm - normalized peak array to the reference m/z list
        """
        # Dynamically load scaler from checkpoint folder
        checkpoint_dir = "/".join(self.checkpoint.split("/")[:-1])  # Get directory of checkpoint
        scaler_path = f"{checkpoint_dir}/scaler.joblib"
        if pathlib.Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
            SimpleLogger.info(f"Loaded scaler from {scaler_path}")
            peaks_norm = scaler.transform(peaks)
            peaks_norm = peaks_norm.clip(0, 1) 
        else:
            SimpleLogger.warning("No fitted MinMax Scaler. Using ion minmax to normalize.")
            max_ion_int = np.max(peaks, axis=0)
            min_ion_int = np.min(peaks, axis=0)
            peaks_norm = np.empty(peaks.shape)
            for i in range(peaks.shape[1]):
                    if max_ion_int[i]!=min_ion_int[i]:
                            peaks_norm[:,i] = (peaks[:,i]-min_ion_int[i])/(max_ion_int[i]-min_ion_int[i])
        return peaks_norm
    
    
    def sample_normalize_peaks(self, peaks):
        """
        Normalize each spectrum (row) independently.
        Inputs:
            peaks - peak array of slide
        Output:
            peaks_norm - normalized peak array to the reference m/z list
        """
        # Compute row-wise min and max
        row_min = peaks.min(axis=1, keepdims=True)  # Shape: (N, 1)
        row_max = peaks.max(axis=1, keepdims=True)  # Shape: (N, 1)
        row_range = row_max - row_min

        # Avoid division by zero
        row_range[row_range == 0] = 1

        # Normalize each spectrum independently
        peaks_norm = (peaks - row_min) / row_range

        return peaks_norm
    
    
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
    
    
    def find_best_legend_position(self, mask):
        """
        Finds the best position for the legend based on white space (NaN values).
        Returns a position tuple (x, y) in normalized figure coordinates.
        """
        nan_mask = np.isnan(mask)
        height, width = nan_mask.shape

        # Divide image into 4 quadrants and count NaN pixels in each
        quad_counts = {
            'upper_left': np.sum(nan_mask[:height//2, :width//2]),
            'upper_right': np.sum(nan_mask[:height//2, width//2:]),
            'lower_left': np.sum(nan_mask[height//2:, :width//2]),
            'lower_right': np.sum(nan_mask[height//2:, width//2:])
        }

        # Sort quadrants by most empty space
        best_quad = max(quad_counts, key=quad_counts.get)

        # Define corresponding positions in figure coordinates
        positions = {
            'upper_left': (0.05, 0.95),
            'upper_right': (0.95, 0.95),
            'lower_left': (0.05, 0.05),
            'lower_right': (0.95, 0.05)
        }

        return positions[best_quad]

    def visualize_predictions(self, predictions, dim_y, dim_x, save_path, mask_path=None):
        """
        Create plot of pixel level predictions from model.

        Inputs:
            predictions - array of predictions for each peak/pixel
            dim_y - Y dimension of original MSI (int)
            dim_x - X dimension of original MSI (int)
            save_path - location to save plot
        """

        preds_img = predictions.reshape((dim_y, dim_x), order='C')

        if mask_path is not None:
            mask_img = Image.open(mask_path).convert("L")
            mask_array = np.array(mask_img)
            binary_mask = (mask_array > 0).astype(int)
            preds_img = np.where(binary_mask == 1, preds_img, np.nan)

        # Define custom colormap (0 = non-cancerous -> green, 1 = cancerous -> red)
        cmap_custom = ListedColormap(["green", "red"])

        fig, ax = plt.subplots(figsize=(6,6))  # Set figure size
        im = ax.imshow(preds_img, cmap=cmap_custom, vmin=0, vmax=1) 
        ax.axis('off')

        # Create legend manually
        legend_patches = [
            mpatches.Patch(color='green', label='Non-cancerous'),
            mpatches.Patch(color='red', label='Cancerous')
        ]

        # Find best position in white space
        legend_x, legend_y = self.find_best_legend_position(preds_img)

        # Add legend dynamically at best location
        ax.legend(handles=legend_patches, loc='center', bbox_to_anchor=(legend_x, legend_y), frameon=False)

        # Ensure the layout prevents legend from overlapping
        plt.tight_layout()

        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()

        
    def visualize_embeddings(self, embeddings, dim_y, dim_x, save_path, dimension=None, metric='pca', mask_path=None):
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
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        # Reshape embeddings into the original image shape
        if metric == "pca":
            pca = PCA(n_components=3)
            embeddings_pca = pca.fit_transform(embeddings)
            print(f"PCA completed. Explained variance ratio: {pca.explained_variance_ratio_}")

            # Reshape to RGB image (dim_y, dim_x, 3)
            pca_image = embeddings_pca.reshape((dim_y, dim_x, 3), order='C')

            # Normalize each channel to [0, 1] for proper visualization
            pca_image = (pca_image - np.min(pca_image, axis=(0, 1))) / (
                np.ptp(pca_image, axis=(0, 1)) + 1e-6
            )

            # Save RGB image
            plt.figure(figsize=(10, 8))
            plt.imshow(pca_image)
            plt.title("RGB PCA Visualization (3 Components)")
            plt.axis('off')
            plt.savefig(f"{save_path}_pca_rgb.png", bbox_inches='tight', dpi=600)
            plt.close()

            print(f"RGB PCA visualization saved at {save_path}_pca_rgb.png")
            
            return pca_image, metric
        
        else:
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
            emb_img = (emb_img - np.min(emb_img)) / (np.max(emb_img) - np.min(emb_img))

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

            return emb_img, metric
    
    
    def cluster_embeddings(self, emb_img, metric, dim_y, dim_x, save_path, n_clusters=5):
        """
        Cluster embeddings or PCA-reduced embeddings to generate a pixel-wise segmentation map.

        Inputs:
            emb_img - array of embeddings for each pixel (NumPy array of shape [num_pixels, num_dimensions]).
            dim_y - Y dimension of the original MSI (int).
            dim_x - X dimension of the original MSI (int).
            save_path - Path to save the clustering result as an image.
            n_clusters - Number of clusters for segmentation.
            method - Clustering method ('kmeans' or 'dbscan').
            metric - Whether to cluster based on PCA-reduced embeddings ('pca') or raw embeddings.
        """
        # Reshape for clustering (flatten to [num_pixels, num_features])
        if metric == 'pca':
            flattened_data = emb_img.reshape(-1, 3)
        else:
            flattened_data = np.nan_to_num(emb_img, nan=0.0, posinf=1e6, neginf=-1e6)

        # Perform clustering
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        
        cluster_labels = clustering_model.fit_predict(flattened_data)

        # Reshape cluster labels back into image dimensions
        cluster_image = cluster_labels.reshape((dim_y, dim_x))

        # Visualize the clustering result
        plt.figure(figsize=(10, 8))
        plt.imshow(cluster_image, cmap='tab10')
        plt.colorbar(label="Cluster")
        plt.title(f"Pixel-wise Clustering ({method.capitalize()})")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()

        print(f"Clustering complete. Results saved to {save_path}")
        
        
    def deploy(self, desi_data_path, save_path, normalize, visualize_embeddings=False, cluster=False, mask_path=None, n_clusters=2, dimension=None, metric='mean'):
        """
        Deploy module to visualize model performance
        Inputs:
            desi_data_path - path to input DESI data file (text or csv)
            normalization - method for normalization
            save_path - location to save output plot
        """
        if desi_data_path.endswith("txt"):
            peaks, mz, dim_y, dim_x = self.DESI_txt2numpy(desi_data_path)
        elif desi_data_path.endswith("csv"):
            peaks, mz, dim_y, dim_x = self.DESI_csv2numpy(desi_data_path)
        else:
            raise TypeError(f"Model deployment only supports .csv and .txt input files")
        
        # align peaks to the model referecne m/z list
        aligned_peaks, aligned_mz = self.peak_alignment_to_reference(mz, peaks, self.mz_ref)
        aligned_peaks = np.nan_to_num(aligned_peaks)
        
        # pre-processing
        aligned_peaks = self.tic_normalize(aligned_peaks)
        if normalize == "MinMaxNormalize":
            aligned_peaks_norm = self.minmax_normalize_peaks(aligned_peaks)
            predictions, embeddings = self.predict(aligned_peaks_norm)
        elif normalize == "SampleNormalize":
            aligned_peaks_norm = self.sample_normalize_peaks(aligned_peaks)
            predictions, embeddings = self.predict(aligned_peaks_norm)
        else:
            predictions, embeddings = self.predict(aligned_peaks)
        
        self.visualize_predictions(predictions, dim_y, dim_x, save_path, mask_path)
        if visualize_embeddings:
            emb_img, metric = self.visualize_embeddings(embeddings, dim_y, dim_x, save_path=f"{save_path}_embeddings", 
                                                        mask_path=mask_path, metric=metric, dimension=dimension)
        if cluster:
            if metric == 'pca':
                self.cluster_embeddings(emb_img, metric, dim_y, dim_x, save_path=f"{save_path}_clustering")
            else:
                self.cluster_embeddings(embeddings, metric, dim_y, dim_x, save_path=f"{save_path}_clustering")
        print(f"Deployment complete. Results saved to {save_path}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a trained DESI model for segmentation.")
    parser.add_argument("--checkpoint", type=str, default="aligned_nockpt_13/checkpoint_17_auroc=0.9994.pt", help="Path to the model checkpoint file.")
    parser.add_argument("--normalize", type=str, help="Peak normalization method")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output segmentation map.")
    parser.add_argument("--desi_data_path", type=str, default="raw data/2021 03 31 colon 1258561-2 Analyte 7.txt", help="Path to the DESI text file for input data.")
    parser.add_argument("--mz_ref", type=str, default="mz_ref/900_peaks.npy", help="Path to the m/z reference file.")
    parser.add_argument("--base_model", type=str, default="laion/clap-htsat-unfused", help="Base model name.")
    parser.add_argument("--projection_head", action="store_true", default=True, help="Use projection head.")
    parser.add_argument("--random_init", action="store_true", default=False, help="Randomly initialize the model.")
    parser.add_argument("--prediction_head", type=int, nargs='+', default=[128, 2], help="Prediction head dimensions.")
    parser.add_argument("--visualize_embeddings", type=bool, default=False, help="Whether to visualize embeddings.")
    parser.add_argument("--cluster", type=bool, default=False, help= "Whether to cluster embeddings")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to mask for masked deployment")
    parser.add_argument("--dimension", type=int, default=None, help="Specific embedding dimension to visualize, if none computes summary metric")
    parser.add_argument("--metric", type=str, default="mean", help="Summary metric for the embeddings. Options: 'L2', 'mean', 'max', 'pca'.")
    
    # Parse arguments
    args = parser.parse_args()

    # Load m/z reference
    mz_ref = list(np.load(args.mz_ref, allow_pickle=True))

    class_order = ['non-cancerous', 'cancerous']

    # Initialize and deploy
    deployment = DESIModelDeployment(
        base_model=args.base_model,
        projection_head=args.projection_head,
        random_init=args.random_init,
        prediction_head=args.prediction_head,
        checkpoint=args.checkpoint,
        mz_ref=mz_ref,
        class_order=class_order,
    )
    deployment.deploy(
        desi_data_path=args.desi_data_path,
        save_path=args.save_path,
        normalize=args.normalize,
        cluster=args.cluster,
        visualize_embeddings=args.visualize_embeddings,
        mask_path=args.mask_path,
        dimension=args.dimension,
        metric=args.metric
    )

