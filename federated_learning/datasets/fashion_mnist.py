# import pickle
# import numpy as np
# import torch
# import sys
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from loguru import logger
# from .dataset import Dataset

# # Configure Logger
# logger.remove()

# # Add handler for terminal output
# logger.add(sys.stdout, format="{time} | {level} | {message}", level="DEBUG")

# # Add handler for saving logs to a file
# logger.add("fashion_mnist_log.txt", format="{time} | {level} | {message}", level="DEBUG")

# class FashionMNISTDataset(Dataset):
#     def __init__(self, args):
#         super(FashionMNISTDataset, self).__init__(args)

#     def load_train_dataset(self):
#         self.get_args().get_logger().debug("Loading Fashion MNIST train data")

#         train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#         train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

#         train_data = self.get_tuple_from_data_loader(train_loader)

#         self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")

#         return train_data

#     def load_test_dataset(self):
#         self.get_args().get_logger().debug("Loading Fashion MNIST test data")

#         test_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#         test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

#         test_data = self.get_tuple_from_data_loader(test_loader)

#         self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")

#         return test_data

#     def flip_labels(self, dataset, flip_percentage=0.2, flip_pairs=[(5, 3)]):
#         """
#         Flips a given percentage of labels based on the provided flip pairs.
#         """
#         targets = np.array(dataset.targets)

#         for source_class, target_class in flip_pairs:
#             class_indices = np.where(targets == source_class)[0]
#             num_to_flip = int(len(class_indices) * flip_percentage)

#             if num_to_flip == 0:
#                 logger.warning(f"No samples from Class {source_class} to flip.")
#                 continue

#             flip_indices = np.random.choice(class_indices, num_to_flip, replace=False)
#             targets[flip_indices] = target_class

#             logger.warning(f"⚠️ Flipped {num_to_flip} samples from Class {source_class} → Class {target_class}")

#         dataset.targets = targets.tolist()

# # Save the processed dataset
# def save_dataset(dataset, file_path):
#     with open(file_path, 'wb') as f:
#         pickle.dump(dataset, f)
#     logger.success(f"🎉 Saved dataset to {file_path}")

# # Initialize and save Fashion-MNIST dataset
# def main(args):
#     fashion_mnist_dataset = FashionMNISTDataset(args)

#     # Load and save the train dataset
#     train_data = fashion_mnist_dataset.load_train_dataset()
#     save_dataset(train_data, "fashion_mnist_train_data.pickle")

#     # Load and save the test dataset
#     test_data = fashion_mnist_dataset.load_test_dataset()
#     save_dataset(test_data, "fashion_mnist_test_data.pickle")

#     logger.success("✅ Dataset processing and saving complete.")



import pickle
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loguru import logger
from .dataset import Dataset

# Configure Logger
logger.remove()
logger.add("label_flipping_log.txt", format="{time} | {level} | {message}", level="DEBUG")

# Add handler for terminal output
logger.add(sys.stdout, format="{time} | {level} | {message}", level="DEBUG")

class FashionMNISTDataset(Dataset):

    def __init__(self, args):
        super(FashionMNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST train data")

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)

        # Apply Label Flipping
        self.flip_labels(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST train data")
        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST test data")

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        test_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)

        # Apply Label Flipping (if required for test set)
        #self.flip_labels(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Fashion MNIST test data")
        return test_data

    def flip_labels(self, dataset, flip_percentage=0.8, source_class=5, target_class=3):
        """
        Flips a given percentage of labels from source_class to target_class in the dataset.
        """
        targets = np.array(dataset.targets)

        # Find samples of the source class
        class_indices = np.where(targets == source_class)[0]
        num_to_flip = int(len(class_indices) * flip_percentage)

        # Randomly select samples to flip
        flip_indices = np.random.choice(class_indices, num_to_flip, replace=False)

        # Perform label flipping
        targets[flip_indices] = target_class
        dataset.targets = targets.tolist()

        # Log label flipping information
        logger.warning(f"⚠️ Flipped {num_to_flip} samples from Class {source_class} → Class {target_class}")

# Save the processed dataset
def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
    logger.success(f"🎉 Saved dataset to {file_path}")

# Initialize and save Fashion-MNIST dataset
def main(args):
    fashion_mnist_dataset = FashionMNISTDataset(args)

    # Load and save the train dataset
    train_data = fashion_mnist_dataset.load_train_dataset()
    save_dataset(train_data, "fashion_mnist_train_data.pickle")

    # Load and save the test dataset
    test_data = fashion_mnist_dataset.load_test_dataset()
    save_dataset(test_data, "fashion_mnist_test_data.pickle")

    logger.success("✅ Dataset processing and saving complete.")
