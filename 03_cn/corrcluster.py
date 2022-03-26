"""Functions for correlation based complete linkage clustering."""
import numpy as np
import scipy.stats as stat
import scipy.cluster as cl
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


class CorrelationBasedClustering:
    """Complete linkage clustering of correlation matrix.

    Args:
    -----
    correlation: np.ndarray (n, n)
        Adjencency matrix of spearman correlations
    pvalue: np.ndarray (n, n)
        P-value matrix of pairwise significant test of spearman correlation
    confidence: float
        Confidence level at which the threshold is obtained
    """

    def __init__(self, correlation, pvalue, confidence, linkage="complete"):
        self.corr = correlation
        self.pvalue = pvalue
        self.confidence = confidence
        self.threshold, self.distance, self.positiv_corr = self.get_distance()
        self.label, self.model = self.complete_linkage_cluster(
            self.threshold, linkage)
        self.num_clusters = np.max(self.label)+1

    def get_distance(self):
        """Get correlation and distance threshold for a given confidence level.

        Note: only positive correlations are considered here

        Return:
        -----
        threshold: float
            Threshold where the clustering is stopped
        distance: np.ndarray (n, n)
            Distance matrix
        corr_pos: np.ndarray (n, n)
            Correlation matrix with only positive correlations
        """
        # get only positive correlations
        mask_corr = np.ma.masked_where((self.corr < 0), self.corr)
        corr_pos = mask_corr.data.copy()
        corr_pos[mask_corr.mask] = -1.0

        # get distance matrix
        distance = np.arccos(corr_pos)

        # consider only correlations with corresponding pvalues smaller than (1-confidence)
        mask_pvalue = np.ma.masked_where(
            (self.pvalue > (1-self.confidence)), self.pvalue)
        mask_idx = np.logical_or(mask_corr.mask, mask_pvalue.mask)
        mask_corr.data[mask_idx] = np.NaN

        # get threshold
        idx_min = np.unravel_index(
            np.nanargmin(mask_corr.data), np.shape(mask_corr.data)
        )
        threshold_corr = self.corr[idx_min]
        threshold_dist = distance[idx_min]

        print(f"p-value {self.pvalue[idx_min]}, "
              + f"correlation {threshold_corr}, distance {threshold_dist}")

        return threshold_dist, distance, corr_pos

    def complete_linkage_cluster(self, threshold, linkage="complete"):
        """Complete linkage clustering.
        Return:
        -------
        labels: list (n)
            Cluster label of each datapoint
        model: sklearn.cluster.AgglomerativeClustering
            Complete linkage clustering model
        """
        # create hierarchical cluster
        model = AgglomerativeClustering(
            distance_threshold=threshold, n_clusters=None, compute_full_tree=True,
            affinity='precomputed', connectivity=None, linkage=linkage
        )
        labels = model.fit_predict(self.distance)
        print(
            f"Found {np.max(labels)+1} clusters for the given threshold {threshold}.")
        return labels, model

    def plot_dendrogram(self, **kwargs):
        """Plot dendogram for a given hierarchical clustering model.
        Args:
        -----
        kwargs: see scipy.cluster.hierarchy.dendrogram args

        Return:
        -------
        linkage_matrix: np.ndarray (n, 2)
            Full linkage matrix of model
        """
        # create the counts of samples under each node
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_,
                                          counts]).astype(float)
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        return linkage_matrix

    def mean_cluster_correlation(self, correlation, linkage_matrix, dist=1.0):
        """Average correlation per cluster for a given distance.
        Args:
        -----
        correlation: np.ndarray (n, n)
            Adjencency matrix of spearman correlations
        linkage_matrix: np.ndarray (n, 2)
            Full linkage matrix of model
        dist: float
            Distance where to cut the dendogram represented by the linkage matrix

        Return:
        -------
        mean_corr: list
            Average correlation per cluster
        num_clusters: int
            Number of clusters at the given distance
        """

        level = cl.hierarchy.fcluster(
            linkage_matrix, t=dist, criterion='distance')
        num_clusters = np.max(level)+1
        mean_corr = []
        for c in range(1, num_clusters):
            idx_class = np.where(level == c)[0]
            class_corr = 0
            for i in idx_class:
                class_corr += np.sum(self.corr[i, idx_class])

            # mean class correlation
            class_corr = class_corr / (len(idx_class) ** 2)
            mean_corr.append(class_corr)

        assert len(mean_corr) == np.max(level)
        print(
            f"{np.max(level)} clusters for distance {dist} with median {np.median(mean_corr)} ")

        return mean_corr, num_clusters

    def median_correlation(self, linkage_matrix, thresholds):
        """Median of average correlation of clusters at the given thresholds.
        Args:
        -----
        linkage_matrix: np.ndarray

        thresholds: list
            Distance thresholds at which to compute the mean correlation within clusters

        Returns:
        --------
        median_correlation: list

        num_clusters: list

        """
        median_correlation = []
        num_clusters = []
        for t in thresholds:
            mean_cluster_corr, clusters = self.mean_cluster_correlation(
                self.positiv_corr, linkage_matrix, dist=t
            )
            # median of average class correlation
            median_correlation.append(np.median(mean_cluster_corr))
            num_clusters.append(clusters)

        return median_correlation, num_clusters

    def time_series(self):
        """Get points with highest correlation to all other points in a cluster.

        Returns:
        --------
        idx_ts: np.ndarray (num_clusters,)
            Index of points for time-series with highest correlation
        cluster: np.ndarray (num_clusters,)
            Corresponding cluster label
        """
        # set negative correlations to Nan
        mask_corr = np.ma.masked_where((self.corr < 0.0), self.corr)
        mask_corr.data[mask_corr.mask] = np.NaN
        pos_corr = mask_corr.data

        # Get time series for each cluster
        ts_attr = {'idx_lat': [], 'idx_lon': [],
                   'lat': [], 'lon': []}
        cluster = []
        idx_ts = []
        for k in range(0, self.num_clusters):
            idx_cluster, = np.where(self.label == k)
            sum_corr = np.nansum(pos_corr[idx_cluster, :], axis=1)
            assert len(idx_cluster) == len(sum_corr)
            cluster.append(k)
            idx_ts.append(idx_cluster[np.argmax(sum_corr)])

        return np.array(idx_ts), np.array(cluster)

    def get_threshold(self):
        return self.threshold

    def get_model(self):
        return self.model

    def get_label(self):
        return self.label
