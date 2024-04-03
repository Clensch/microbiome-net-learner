# author: Michael Hüppe
# date: 15.12.2023
# project: biostat
# external

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    """
    Create a simple sample dataset
    """
    def __init__(self, num_samples, num_features, random_seed=42):
        self.data, self.labels = make_classification(
            n_samples=num_samples, n_features=num_features, n_informative=num_features // 2,
            n_redundant=0, n_clusters_per_class=1, random_state=random_seed
        )
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class GenomDataset(Dataset):
    """
    Manage genome dataset
    """
    def __init__(self, path: str, keepFeaturesRatio: float = None):
        self._keepFeaturesRatio = keepFeaturesRatio
        # read in data
        data = pd.read_csv(path)
        data = self._clean(data)
        # get the labels and remove them from the data
        labels = data["host_phenotype"].values
        data = data.drop('host_phenotype', axis=1)
        self.columns = data.columns
        data = data.values
        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        self.num_classes = len(np.unique(labels))
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)

        # Normalize each column
        normalized_data = (data - mean) / std_dev
        self.data = torch.FloatTensor(np.nan_to_num(normalized_data, nan=0))
        self.labels = torch.LongTensor(labels)

    @classmethod
    def _clean(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data to be used for training
        :param data: dataframe containing genome information
        :return: cleaned data frame
        """
        # age, bmi, gender contain NaN values
        gender_mapping = {'male': 0.0, 'female': 1.0}
        data['gender'] = data['gender'].map(gender_mapping)

        # Change host_phenotype to numeric; CRC = 0, healthy = 1
        host_phenotype_mapping = {'CRC': 0.0, 'healthy': 1.0}
        data['host_phenotype'] = data['host_phenotype'].map(host_phenotype_mapping)

        # Delete ID, study_accession, country, age.1 (dupplication of age)
        data = data.drop('Unnamed: 0', axis=1)
        data = data.drop('study_accession', axis=1)
        data = data.drop('country', axis=1)
        if "age.1" in data.columns:
            data = data.drop('age.1', axis=1)

        # Fill every NaN value --> Take the mean of the column, since the only values affected are in age, bmi, gender
        columns_with_nan_values = ['age', 'bmi', 'gender']
        for column in columns_with_nan_values:
            mean_value = data[column].mean()
            data[column].fillna(mean_value, inplace=True)

        return data

    @classmethod
    def _normalize(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the input data
        :param data: dataframe containing genome data
        :return: Normalized dataframe
        """
        # Log Normalization
        features_to_norm = [col for col in data.columns if col not in ['age', 'host_phenotype', 'bmi', 'gender']]
        data[features_to_norm] = data[features_to_norm].apply(lambda x: np.log1p(x))

        # Z-Score Normalization
        scaler = StandardScaler()
        data[features_to_norm] = scaler.fit_transform(data[features_to_norm])
        return data

    def _reduceFeatures(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce the number of features using Principal Component Analysis
        :param data: data to reduce features of
        :return: reduced data
        """
        pca = PCA()
        principal_components = pca.fit_transform(data)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        pca = PCA(n_components=np.argmax(cumulative_variance_ratio >= self._keepFeaturesRatio) + 1)
        data = pca.fit_transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
