import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_olivetti_faces,
    load_breast_cancer,
    load_digits,
    load_iris,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor
from ucimlrepo import fetch_ucirepo

from .doh import DoH
from .har import HARDataset
from .kaggle_utils import (
    check_kaggle_credentials,
    download_kaggle_dataset,
)

logging.getLogger("lightning").setLevel(0)


class DatasetHandler(pl.LightningDataModule):
    DEFAULT_BATCH_SIZE = 32
    dataset_to_default_batch_size = {
        "iris": 150,
        "digits": 512,
        "digits2d": 512,
        "breast_cancer": 512,
        "olivetti_faces": 150,
        "cifar10": 512,
        "cifar10gray": 512,
        "cifar100": 128,
        "emnist": 128,
        "fer2013": 128,
        "fmnist": 512,
        "mnist": 512,
        "census_income": 512,
        "covert": 512,
        "har": 512,
        "doh": 2048,
        "mnist1d": 512,
    }

    def __init__(
        self,
        dataset_name,
        root="./data",
        k_fold=5,
        transform=None,
        batch_size=None,
        test_split_size=0.2,
        max_classes=None,
        binary_split: Optional[bool] = None,
        max_samples: Optional[int] = None,
        balance_method: Optional[str] = None,
    ):
        super().__init__()
        if dataset_name.startswith("binary:"):
            dataset_name = dataset_name[7:]
            binary_split = True
        else:
            binary_split = binary_split if binary_split is not None else False

        self.dataset_name = dataset_name
        self.root = root
        self.k_fold = k_fold
        self.transform = transform if transform is not None else ToTensor()
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = self.dataset_to_default_batch_size.get(
                dataset_name, self.DEFAULT_BATCH_SIZE
            )
        self.test_split_size = test_split_size
        self.max_classes = max_classes
        self.binary_split = binary_split
        self.max_samples = max_samples
        self.balance_method = balance_method

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.folds = None
        self.input_size = None
        self.num_classes = None

        self.sklearn_dataset = ["iris", "digits", "digits2d", "breast_cancer", "olivetti_faces"]
        self.torchvision_dataset = [
            "cifar10",
            "cifar10gray",
            "cifar100",
            "emnist",
            "fer2013",
            "fmnist",
            "mnist",
        ]
        self.uci_machine_learning_repo = ["census_income", "covert"]
        self.custom_dataset = ["har", "doh", "mnist1d"]

        self.available_datasets = (
            self.sklearn_dataset
            + self.torchvision_dataset
            + self.uci_machine_learning_repo
            + self.custom_dataset
        )

        self.shape = None

        self.fold_index = 0

    @property
    def batched_shape(self):
        return self.batch_size, self.input_size

    @property
    def input_shape(self) -> Sequence[int]:
        return tuple(self.shape)

    @property
    def output_shape(self) -> Sequence[int]:
        return (self.num_classes,)

    @property
    def inputs(self):
        dataloader = DataLoader(self.dataset, batch_size=len(self.dataset))
        x_train, y_train = next(iter(dataloader))
        return x_train

    @property
    def targets(self):
        dataloader = DataLoader(self.dataset, batch_size=len(self.dataset))
        x_train, y_train = next(iter(dataloader))
        return y_train

    def load_dataset(self, dataset_name: str) -> Dataset:
        dataset_name = dataset_name.lower()
        if dataset_name in self.sklearn_dataset:
            self.dataset = self._sklearn_load_dataset(dataset_name)
        elif dataset_name in self.torchvision_dataset:
            self.dataset = self._torchvision_load_dataset(dataset_name)
        elif dataset_name in self.uci_machine_learning_repo:
            self.dataset = self._load_uci_machine_learning_repo(dataset_name)
        elif dataset_name in self.custom_dataset:
            self.dataset = self._custom_load_dataset(dataset_name)
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported, choose from {self.available_datasets}"
            )

        return self.dataset

    def prepare_data(self):
        self.dataset = self.load_dataset(self.dataset_name)
        self.filter_classes()
        if self.max_samples is not None:
            rn_state = np.random.RandomState(0)
            rn_indices = rn_state.choice(
                range(len(self.dataset)), size=self.max_samples, replace=False
            )
            self.dataset = Subset(self.dataset, rn_indices)

        self.set_fold(self.fold_index)
        self.shape = self.dataset[0][0].shape

    def filter_classes(self):
        all_labels = [int(item[1].item()) for item in self.dataset]
        unique_labels = list(set(all_labels))

        if self.binary_split:
            # Perform the binary split
            split_point = len(unique_labels) // 2
            first_class_group = unique_labels[:split_point]
            second_class_group = unique_labels[split_point:]

            # Create a new dataset with binary labels
            new_data = []
            for img, label in self.dataset:
                new_label = 0 if label in first_class_group else 1
                new_data.append((img, new_label))

            # Convert the list to a TensorDataset
            images, labels = zip(*new_data)
            images = torch.stack(images)  # Stack images into a single tensor
            labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

            self.dataset = torch.utils.data.TensorDataset(images, labels)
            self.num_classes = 2
        elif self.max_classes is not None and self.max_classes < len(unique_labels):
            # Select only the first max_classes unique labels
            selected_labels = unique_labels[: self.max_classes]
            filtered_indices = [
                i for i, label in enumerate(all_labels) if label in selected_labels
            ]
            self.dataset = Subset(self.dataset, filtered_indices)
            self.num_classes = len(selected_labels)
        else:
            self.num_classes = len(unique_labels)

    def balance_dataset(self, X, y, method="under"):
        if method == "under":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler()
        elif method == "over":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler()
        elif method == "smoteenn":
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN()
        else:
            raise ValueError(
                "Invalid method for balancing dataset. Choose 'under', 'over', or 'smoteenn'."
            )

        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    def _custom_load_dataset(self, dataset_name: str) -> torch.utils.data.Dataset:
        if dataset_name == "har":
            dataset = HARDataset(root=self.root)
        elif dataset_name == "doh":
            doh = DoH()
            doh.prepare_data()
            dataset = doh.dataset
        elif dataset_name == "mnist1d":
            from mnist1d.data import get_dataset_args, make_dataset

            defaults = get_dataset_args()
            data = make_dataset(defaults)
            x, y, t = data["x"], data["y"], data["t"]

            # Convert numpy arrays to PyTorch tensors
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            # Create a TensorDataset
            dataset = TensorDataset(x_tensor, y_tensor)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported.")
        return dataset

    def _load_doh_dataset(self):
        # Path to your dataset files
        data_paths = [
            f"{self.root}/l1-doh.csv",
            f"{self.root}/l1-nondoh.csv",
            f"{self.root}/l2-benign.csv",
            f"{self.root}/l2-malicious.csv",
        ]

        # Load and concatenate data
        data_frames = [pd.read_csv(path) for path in data_paths]
        df = pd.concat(data_frames, ignore_index=True)

        # Preprocessing steps
        df.replace({"?": np.nan}, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Drop columns that are not features (customize this as per your needs)
        df.drop(
            columns=[
                "SourceIP",
                "DestinationIP",
                "SourcePort",
                "DestinationPort",
                "TimeStamp",
            ],
            inplace=True,
        )

        # Convert categorical labels to numeric
        df["Label"] = df["Label"].astype("category").cat.codes

        # Separate features and target
        X = df.drop("Label", axis=1).values
        y = df["Label"].values

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset

    def _load_uci_machine_learning_repo(
        self, dataset_name: str
    ) -> torch.utils.data.Dataset:
        # Fetch the dataset based on the provided dataset name
        if dataset_name == "census_income":
            dataset = fetch_ucirepo(id=20)
        elif dataset_name == "covert":
            dataset = fetch_ucirepo(id=31)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported.")

        # Extract features and targets
        X = dataset.data.features.copy()
        y = dataset.data._targets.copy()

        # Replace missing values represented by "?" with NaN and drop those rows
        X.replace("?", np.nan, inplace=True)
        X.dropna(inplace=True)

        # Align target variable y with the filtered features X
        y = y.loc[X.index]

        # Separate columns into categorical and numerical
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        # Create transformers for the data
        # Binary encoder for categorical features with 2 categories
        binary_transformer = OrdinalEncoder()

        # OneHot encoder for categorical features with more than 2 categories
        onehot_transformer = OneHotEncoder(
            sparse_output=False
        )  # set sparse_output to False to avoid sparse matrix

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), numerical_cols),
                (
                    "bin",
                    binary_transformer,
                    [col for col in categorical_cols if X[col].nunique() == 2],
                ),
                (
                    "onehot",
                    onehot_transformer,
                    [col for col in categorical_cols if X[col].nunique() > 2],
                ),
            ],
            remainder="passthrough",  # Apply MinMaxScaler to all other columns
        )

        # Apply preprocessing and convert to numpy array
        X_preprocessed = preprocessor.fit_transform(X)

        # Convert to tensors
        X_tensor = torch.tensor(X_preprocessed, dtype=torch.float32)
        y = y.squeeze()  # Ensure y is a 1D array
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Update class attributes
        self.input_size = X_tensor.shape[1]
        self.num_classes = len(np.unique(y_encoded))

        return dataset

    def _torchvision_load_dataset(self, dataset_name: str) -> torch.utils.data.Dataset:
        base_target_transform = Compose([lambda y: torch.tensor(y, dtype=torch.long)])
        base_inputs_transform_func = lambda size, *others: Compose([
            ToTensor(),
            # transforms.Normalize(0.0, 1.0),
            # transforms.RandomCrop(32, pad_if_needed=True),  # Randomly crop a 32x32 patch
            # transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
            # transforms.RandomRotation(10),  # Randomly rotate up to 10 degrees
            # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            *others,
        ])

        if dataset_name in ["mnist", "fmnist", "emnist"]:
            mnist_class = None
            dataset_args = {
                "root": self.root,
                "train": True,
                "download": True,
                "transform": base_inputs_transform_func(28),
                "target_transform": base_target_transform,
            }

            if dataset_name == "mnist":
                from torchvision.datasets import MNIST
                mnist_class = MNIST
            elif dataset_name == "fmnist":
                from torchvision.datasets import FashionMNIST
                mnist_class = FashionMNIST
            elif dataset_name == "emnist":
                from torchvision.datasets import EMNIST
                mnist_class = EMNIST
                dataset_args["split"] = (
                    "balanced"  # EMNIST requires the 'split' argument
                )

            # Instantiate train and test datasets with the constructed arguments
            train_dataset = mnist_class(**dataset_args)
            # Update the 'train' key for the test dataset
            dataset_args["train"] = False
            test_dataset = mnist_class(**dataset_args)

            # Combine the datasets
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        elif dataset_name in ["cifar10", "cifar10gray"]:
            from torchvision.datasets import CIFAR10
            transforms_compoenents = []
            if dataset_name == "cifar10gray":
                transforms_compoenents.append(
                    transforms.Grayscale(num_output_channels=1)
                )
            transform = base_inputs_transform_func(32, *transforms_compoenents)
            train_dataset = CIFAR10(
                self.root,
                train=True,
                download=True,
                transform=transform,
                target_transform=base_target_transform,
            )
            test_dataset = CIFAR10(
                self.root,
                train=False,
                download=True,
                transform=transform,
                target_transform=base_target_transform,
            )
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        elif dataset_name == "fer2013":
            from torchvision.datasets import FER2013
            transform = Compose([ToTensor()])
            if not self._check_fer2013_files():
                download_kaggle_dataset("deadskull7/fer2013", "fer2013", self.root)

            dataset = FER2013(
                self.root, transform=transform, target_transform=base_target_transform
            )

        elif dataset_name == "cifar100":
            from torchvision.datasets import CIFAR100
            transform = base_inputs_transform_func(32)
            train_dataset = CIFAR100(
                self.root,
                train=True,
                download=True,
                transform=transform,
                target_transform=base_target_transform,
            )
            test_dataset = CIFAR100(
                self.root,
                train=False,
                download=True,
                transform=transform,
                target_transform=base_target_transform,
            )
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported.")

        return dataset

    def _sklearn_load_dataset(self, dataset_name: str):
        """Load dataset from sklearn."""
        if dataset_name == "iris":
            data = load_iris()
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
        elif dataset_name == "digits":
            data = load_digits()
        elif dataset_name == "digits2d":
            data = load_digits()
            n_feat = data.data.shape[-1]
            data.data = data.data.reshape(data.data.shape[0], 1, int(np.sqrt(n_feat)), int(np.sqrt(n_feat)))
        elif dataset_name == "olivetti_faces":
            data = fetch_olivetti_faces()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported in sklearn.")

        X, y = data.data, data.target
        x_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        X = MinMaxScaler().fit_transform(X)
        X = X.reshape(x_shape)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        if dataset_name == "olivetti_faces":
            n_feat = X.shape[-1]
            X = X.reshape(X.shape[0], 1, int(np.sqrt(n_feat)), int(np.sqrt(n_feat)))

        dataset = torch.utils.data.TensorDataset(X, y)

        return dataset

    def _check_fer2013_files(self) -> bool:
        """Check if FER2013 dataset files are present."""
        fer2013_path = Path(self.root) / "fer2013"
        # Check for any valid dataset file presence
        return any(
            (fer2013_path / file_name).exists()
            for file_name in [
                "fer2013.csv",
                "icml_face_data.csv",
                "train.csv",
                "test.csv",
            ]
        )

    def compute_initial_centroid(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call prepare_data() first.")
        dataloader = DataLoader(self.dataset, batch_size=len(self.dataset))
        x_train, y_train = next(iter(dataloader))
        unique_classes = torch.unique(y_train)
        centroids = []

        for cls in unique_classes:
            centroid = torch.mean(x_train[y_train == cls], dim=0).unsqueeze(0)
            centroids.append(centroid)

        centroids = torch.cat(centroids, dim=0)

        return centroids

    def setup(self, stage=None):
        all_labels = [int(item[1].item()) for item in self.dataset]
        self.num_classes = len(set(all_labels))

        first_item = self.dataset[0][0]
        self.input_size = first_item.numel()

    def create_folds(self):
        # Create train/validation/test splits
        full_length = len(self.dataset)
        train_val_length = int((1 - self.test_split_size) * full_length)
        test_length = full_length - train_val_length

        # Split data into train/validation and test sets
        train_val_indices, test_indices = train_test_split(
            range(full_length), test_size=test_length, random_state=42
        )

        self.test_dataset = Subset(self.dataset, test_indices)
        train_val_dataset = Subset(self.dataset, train_val_indices)

        # If k_fold is set to 1, use a single split
        if self.k_fold == 1:
            # Use a fixed split for train and validation
            train_length = int(0.8 * len(train_val_dataset))  # 80% train, 20% val
            train_indices, val_indices = train_test_split(
                range(len(train_val_dataset)), train_size=train_length, random_state=42
            )

            self.folds = [
                {
                    "train": Subset(train_val_dataset, train_indices),
                    "val": Subset(train_val_dataset, val_indices),
                }
            ]
        else:
            # Create k-fold splits
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
            self.folds = []

            for train_idx, val_idx in kf.split(range(len(train_val_dataset))):
                self.folds.append(
                    {
                        "train": Subset(train_val_dataset, train_idx),
                        "val": Subset(train_val_dataset, val_idx),
                    }
                )

    def set_fold(self, fold_index):
        self.fold_index = fold_index
        if self.folds is None:
            self.create_folds()
        if 0 <= fold_index < self.k_fold:
            self.train_dataset = self.folds[fold_index]["train"]
            self.val_dataset = self.folds[fold_index]["val"]
        else:
            raise ValueError(
                f"Invalid fold index. Must be between 0 and {self.k_fold - 1}"
            )

        if self.balance_method is not None:
            X_train, y_train = zip(
                *[(x.numpy(), y.item()) for x, y in self.train_dataset]
            )
            X_train = np.array(X_train)
            X_train = X_train.reshape(X_train.shape[0], -1)
            y_train = np.array(y_train)

            # Balance the dataset
            X_balanced, y_balanced = self.balance_dataset(
                X_train, y_train, method=self.balance_method
            )

            # Convert balanced data back to tensors
            X_balanced_tensor = torch.tensor(X_balanced, dtype=torch.float32)
            y_balanced_tensor = torch.tensor(y_balanced, dtype=torch.long)

            # Create a balanced dataset
            self.train_dataset = TensorDataset(X_balanced_tensor, y_balanced_tensor)

    def get_input_size(self):
        if self.input_size is None:
            self.setup()
        return self.input_size

    def get_num_classes(self):
        if self.num_classes is None:
            self.setup()
        return self.num_classes

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def __repr__(self):
        return (f""
                f"DatasetHandler("
                f"{self.dataset_name}, "
                f"batch_size={self.batch_size}, "
                f"inputs_shape={self.input_shape}, "
                f"output_shape={self.output_shape}, "
                f"k_fold={self.k_fold}, "
                f"fold={self.fold_index}, "
                f"|train|={len(self.train_dataset)}, "
                f"|val|={len(self.val_dataset)}, "
                f"|test|={len(self.test_dataset)}"
                f")"
                f"")

    def __str__(self):
        return self.__repr__()
