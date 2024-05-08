import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import cub_root, cub_train_augmented, cub_train_push, cub_test
from glob import glob

class CustomCub2011(Dataset):
    """
    W moim podejściu (_load_data_from_folders)
    ta klasa chodzi po ścierzkach (from config) i zbiera:
    'img_id': unikalny numer obrazka
    'filepath': ścieżka do obrazka
    'target': numer klasy (TO JEST DO POPRAWY)
    'is_training_img': 1 jeśli obrazek jest w zbiorze treningowym, 0 jeśli w testowym 
    """

    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root=cub_root, train=True, transform=None, target_transform=None, loader=default_loader, 
                 download=True, data_loading_type='augmented_root', train_path=None, test_path=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train

        self.data_loading_type = data_loading_type
        self.train_path = train_path
        self.test_path = test_path
        
        if download:
            self._download()

        self._load_metadata()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    # def _print_summary(self, title):
    #     print(f"\n\nSummary of data loaded from {title}:")
    #     print(f"Smallest target value: {self.data['target'].min()}")


    def _load_data_from_root(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # self._print_summary("root")

    def _load_data_from_folders(self):
        # Initialize lists to store data
        images = []
        labels = []
        img_ids = []
        folder_types = []

        # Process each folder
        for folder_type, folder_path in [(1, self.train_path), (0, self.test_path)]:
            class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            class_folders = sorted(class_folders, key=lambda x: int(x.split('.')[0]))
            for class_folder in class_folders:
                class_idx = int(class_folder.split('.')[0])
                class_path = os.path.join(folder_path, class_folder)
                for img_file in glob(os.path.join(class_path, '*.jpg')):
                    # Use absolute path for the image
                    absolute_path = os.path.abspath(img_file)

                    images.append(absolute_path)
                    labels.append(class_idx)  # Targets start at 0 by default, so shift to 1
                    img_ids.append(len(images))
                    folder_types.append(folder_type)

        # Combine data into a DataFrame
        self.data = pd.DataFrame({'img_id': img_ids, 'filepath': images, 'target': labels, 'is_training_img': folder_types})

        # Separate train and test data if necessary
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # self._print_summary("folders")

    def _load_metadata(self):
        if self.data_loading_type == 'root':
            try:
                self._load_data_from_root()
            except Exception:
                return False
        elif self.data_loading_type == 'augmented_root':
            self._load_data_from_folders()

    def _check_integrity_root(self):
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _check_integrity_folder(self):
        # Integrity check for 'folder' type
        # Check if the train and test folders exist and are not empty
        if not os.path.isdir(self.train_path) or not os.listdir(self.train_path):
            print(f"Train folder {self.train_path} not found or is empty.")
            return False
        if not os.path.isdir(self.test_path) or not os.listdir(self.test_path):
            print(f"Test folder {self.test_path} not found or is empty.")
            return False
        return True

    def _check_integrity(self):
        if self.data_loading_type == 'root':
            return self._check_integrity_root()
        elif self.data_loading_type == 'augmented_root':
            return self._check_integrity_folder()

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # this one will work properly for loading data from root and from folders:
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cub_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, args=None):

    # Init entire training set
    whole_training_set = CustomCub2011(root=cub_root, train=True, transform=train_transform, \
                                        download=False, data_loading_type=args.data_loading_type, \
                                        train_path=cub_train_augmented, test_path=cub_test)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Init entire PUSH training set
    whole_push_training_set = CustomCub2011(root=cub_root, train=True, transform=train_transform, \
                                        download=False, data_loading_type=args.data_loading_type, \
                                        train_path=cub_train_push, test_path=cub_test)

    # Get labelled training PUSH set which has subsampled classes, then subsample some indices from that
    train_push_dataset_labelled = subsample_classes(deepcopy(whole_push_training_set), include_classes=train_classes)

    # Split into training PUSH and validation sets
    train_push_idxs, val_push_idxs = get_train_val_indices(train_push_dataset_labelled)
    train_push_dataset_labelled_split = subsample_dataset(deepcopy(train_push_dataset_labelled), train_push_idxs)
    val_push_dataset_labelled_split = subsample_dataset(deepcopy(train_push_dataset_labelled), val_push_idxs)
    val_push_dataset_labelled_split.transform = test_transform

    # Get unlabelled PUSH data    
    unlabelled_push_indices = set(whole_push_training_set.uq_idxs) - set(train_push_dataset_labelled.uq_idxs)
    train_push_dataset_unlabelled = subsample_dataset(deepcopy(whole_push_training_set), np.array(list(unlabelled_push_indices)))

    # Get test set for all classes
    test_dataset = CustomCub2011(root=cub_root, train=False, transform=test_transform, \
                                        download=False, data_loading_type=args.data_loading_type, \
                                        train_path=cub_train_augmented, test_path=cub_test)
    test_dataset_seen = subsample_classes(deepcopy(test_dataset), include_classes=train_classes)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    train_push_dataset_labelled = train_push_dataset_labelled_split if split_train_val else train_push_dataset_labelled

    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    val_push_dataset_labelled = val_push_dataset_labelled_split if split_train_val else None
    
    print("Lens: train_labelled: {}, train_unlabelled: {}, val: {}, test: {}, test_seen: {}".format(len(train_dataset_labelled), len(train_dataset_unlabelled), len(val_dataset_labelled) if split_train_val else 0, len(test_dataset), len(test_dataset_seen)))
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'train_push_labelled': train_push_dataset_labelled,
        'train_push_unlabelled': train_push_dataset_unlabelled,
        'val': val_dataset_labelled,
        'val_push': val_push_dataset_labelled,
        'test': test_dataset,
        "test_seen": test_dataset_seen
    }

    return all_datasets

if __name__ == '__main__':

    x = get_cub_datasets(None, None, split_train_val=False,
                         train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')