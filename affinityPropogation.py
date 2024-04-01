import numpy as np
from sklearn.metrics import euclidean_distances
class AffinityPropagationClustering:
    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, preference=None, random_state=0):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.random_state = random_state

    @staticmethod
    def _affinity_propagation_inner(similarity_matrix, preference, convergence_iter, max_iter,
                                   damping, random_state):
        rng = np.random.RandomState(random_state)
        n_samples = similarity_matrix.shape[0]
        samples_indexes = np.arange(n_samples)

        # place preference on the diagonal of similarity matrix
        np.fill_diagonal(similarity_matrix, preference)

        # initialize availability and responsibility matrix
        availability_matrix = np.zeros((n_samples, n_samples))
        responsibility_matrix = np.zeros((n_samples, n_samples))
        exemplars_convergence_matrix = np.zeros((n_samples, convergence_iter))

        for iter in range(max_iter):
            temp_matrix = availability_matrix + similarity_matrix   # compute responsibilities
            max_indexes = np.argmax(temp_matrix, axis=1)
            max_values = np.max(temp_matrix, axis=1)
            temp_matrix[samples_indexes, max_indexes] = -np.inf
            second_max_values = np.max(temp_matrix, axis=1)

            # temp_matrix = new_responsibility_matrix
            np.subtract(similarity_matrix, max_values[:, None], temp_matrix)
            max_responsibility = similarity_matrix[samples_indexes, max_indexes] - second_max_values
            temp_matrix[samples_indexes, max_indexes] = max_responsibility

            # damping
            temp_matrix *= 1 - damping
            responsibility_matrix *= damping
            responsibility_matrix += temp_matrix

            # temp_matrix = Rp; compute availabilities
            np.maximum(responsibility_matrix, 0, temp_matrix)
            np.fill_diagonal(temp_matrix, np.diag(responsibility_matrix))

            # temp_matrix = -new_availability_matrix
            temp_matrix -= np.sum(temp_matrix, axis=0)
            diag_availability_matrix = np.diag(temp_matrix).copy()
            temp_matrix.clip(0, np.inf, temp_matrix)
            np.fill_diagonal(temp_matrix, diag_availability_matrix)

            # damping
            temp_matrix *= 1 - damping
            availability_matrix *= damping
            availability_matrix -= temp_matrix

            # check for convergence
            exemplar = (np.diag(availability_matrix) + np.diag(responsibility_matrix)) > 0
            exemplars_convergence_matrix[:, iter % convergence_iter] = exemplar
            n_exemplars = np.sum(exemplar, axis=0)

            if iter >= convergence_iter:
                exemplars_sum = np.sum(exemplars_convergence_matrix, axis=1)
                unconverged = np.sum((exemplars_sum == convergence_iter) +
                                     (exemplars_sum == 0)) != n_samples

                if (not unconverged and (n_exemplars > 0)) or (iter == max_iter):
                    break

        exemplar_indixes = np.flatnonzero(exemplar)
        n_exemplars = exemplar_indixes.size  # number of detected clusters

        if n_exemplars > 0:
            cluster_indices = np.argmax(similarity_matrix[:, exemplar_indixes], axis=1)
            cluster_indices[exemplar_indixes] = np.arange(n_exemplars)  # Identify clusters

            # refine the final set of exemplars and clusters and return results
            for k in range(n_exemplars):
                cluster_members = np.where(cluster_indices == k)[0]
                best_k = np.argmax(np.sum(similarity_matrix[cluster_members[:, np.newaxis],
                                                            cluster_members], axis=0))
                exemplar_indixes[k] = cluster_members[best_k]

            cluster_indices = np.argmax(similarity_matrix[:, exemplar_indixes], axis=1)
            cluster_indices[exemplar_indixes] = np.arange(n_exemplars)
            labels = exemplar_indixes[cluster_indices]

            # Reduce labels to a sorted, gapless, list
            cluster_centers_indices = np.unique(labels)
            labels = np.searchsorted(cluster_centers_indices, labels)
        else:
            cluster_centers_indices = []
            labels = np.array([-1] * n_samples)

        return cluster_centers_indices, labels

    def fit_predict(self, X):
        self.affinity_matrix_ = -euclidean_distances(X, squared=True)

        if self.preference is None:
            self.preference = np.median(self.affinity_matrix_)

        params = (self.affinity_matrix_, self.preference, self.convergence_iter, self.max_iter,
                  self.damping, self.random_state)

        self.cluster_centers_indices_, self.labels_ = self._affinity_propagation_inner(*params)
        self.cluster_centers_ = X[self.cluster_centers_indices_]

        return self.labels_
