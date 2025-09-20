import os.path

import faiss
import numpy as np


class FaissStorage:
    def __init__(self, path):
        # check if path exists
        if not path:
            raise ValueError("Path to FAISS index must be provided.")
        if not os.path.exists(path):
            raise ValueError(f"FAISS index file not found at {path}")
        self.index = faiss.read_index(path)

    @staticmethod
    def create(ids, embeddings, path):
        """
        Create FAISS Index
        :param ids: numpy array of shape (n_samples,) with integer IDs
        :param embeddings: numpy array of shape (n_samples, n_features)
        :param path: path to save the FAISS index
        """
        d = embeddings.shape[1]  # embedding dimension
        index = faiss.IndexFlatIP(d)  # cosine similarity (after normalization)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(embeddings, np.array(ids))
        faiss.write_index(index, path)

        return FaissStorage(path)

    def search(self, embeddings, top_k) -> tuple:
        """
        Search FAISS Index
        :param embeddings: numpy array of shape (n_samples, n_features)
        :param top_k: number of nearest neighbors to retrieve
        :return: distances and indices of the nearest neighbors
        """
        distances, indices = self.index.search(embeddings, top_k)
        return distances, indices
