#import numpy as np 
#import matplotlib as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import numpy as np
import pickle

class UnlearnDatasetSplit:

    available_datasets = ["cifar-10", "coco"]
    possible_splits = ["Train", "Test", "Train_retain", "Train_forget", "Test_retain", "Test_forget"]

    def __init__(self, path: str, dataset: str):
        self.dataset_path = path
        self.dataset_splits = {}
        self.classes = None
        self.n_classes = 0

        # load from path
        self.load_dataset()

        # if could not find datasets in path
        if len(self.dataset_splits) == 0:
            if dataset != None:
                self.get_dataset(dataset)
                print(f'Classes: {self.classes}')
                self.n_classes = len(self.classes)
            

    def save_dataset(self):

        for k in self.dataset_splits.keys():
            with open(f'{self.dataset_path}{k}.pkl', 'wb') as f:
                pickle.dump(self.dataset_splits[k], f)
        
    def load_dataset(self):

        if len(self.dataset_splits) == 0:
            files = [f for f in os.listdir(self.dataset_path) if f.endswith('.pkl')]
            print(files)
            try:
                for file in files:
                    with open(self.dataset_path + file, 'rb') as f:
                        self.dataset_splits[file[:-4]] = pickle.load(f)
            except:
                print("\033[31mError while loading the dataset from the files!\033[0m")
                return None
        
        return self.dataset_splits

    def get_dataset(self, dataset: str):

        if isinstance(dataset, list) or not isinstance(dataset, str):
            print(f"\033[31mInvalid dataset\r\n Please pick between {available_datasets}!\033[0m")
            return None
        
        if dataset == "cifar10":

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            train = datasets.CIFAR10(self.dataset_path, train=True, transform=transform, download=True)
            test = datasets.CIFAR10(self.dataset_path, train=False, transform=transform, download=True)

            self.classes = train.classes
            self.dataset_splits = {"Train": train, "Test": test}

        elif dataset == "coco":
            print('not implemented yet')

        else:
            print(f"\033[31mInvalid method\r\n Please pick between {available_datasets}!\033[0m")
    
    def get_loader(self, k, batchsize: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True):

        try:
            loader = torch.utils.data.DataLoader(self.dataset_splits[k], batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        except:
            print(f"\033[31mCould not load {k}, check if dataset exists!\033[0m")
            return None
        return loader

    def split_dataset(self, mode: str, save: False, **kwargs):

        if mode == "class":

            if 'forget' in kwargs.keys():
                print(kwargs['forget'])

                if isinstance(kwargs['forget'], list):
                    print('is list')

                    if isinstance(kwargs['forget'][0], str):
                        print('is string list')

                        c = []
                        for i, class_name in enumerate(self.classes):
                            if class_name in kwargs['forget']:
                                c.append(i)
                        
                        c = np.array(c)

                    elif isinstance(kwargs['forget'][0], int):
                        print('is int list')

                        if max(kwargs['forget']) < self.n_classes:                        
                            c = np.array(kwargs['forget'])

                        else:
                            print(f"\033[31mClass index out of range\r\n Please insert a valid class index!\033[0m")
                            return None

                    else:
                        print(f"\033[31mClass format not recognized\r\n Please insert a valid class value!\033[0m")
                        return None
                
                    print(f'class is {c}')
                    trainf_mask = np.isin(np.array(self.dataset_splits["Train"].targets), c)
                    testf_mask = np.isin(np.array(self.dataset_splits["Test"].targets), c)
                    
                elif isinstance(kwargs['forget'], str):
                    print('is string')

                    for i, class_name in enumerate(self.classes):
                        if kwargs['forget'] == class_name:
                            c = i
                            break

                    trainf_mask = np.array(self.dataset_splits["Train"].targets) == c
                    testf_mask = np.array(self.dataset_splits["Test"].targets) == c

                elif isinstance(kwargs['forget'], int):
                    print('is int')
                    if kwargs['forget'] < self.n_classes:
                        trainf_mask = np.array(self.dataset_splits["Train"].targets) == kwargs['forget']
                        testf_mask = np.array(self.dataset_splits["Test"].targets) == kwargs['forget']
                    else:
                        print(f"\033[31mClass index outside of available range\r\n Please insert a valid class index!\033[0m")
                        return None

                else:
                    print(f"\033[31mInvalid class\r\n Please insert a valid class or class index!\033[0m")
                    return None

                train_idx = np.array(range(len(self.dataset_splits["Train"])))
                test_idx = np.array(range(len(self.dataset_splits["Test"])))

                train_f_idx = train_idx[trainf_mask]
                train_r_idx = train_idx[~trainf_mask]
                test_f_idx = test_idx[testf_mask]
                test_r_idx = test_idx[~testf_mask]

                train_retain = Subset(self.dataset_splits["Train"], train_r_idx)
                train_forget = Subset(self.dataset_splits["Train"], train_f_idx)

                test_retain = Subset(self.dataset_splits["Test"], test_r_idx)
                test_forget = Subset(self.dataset_splits["Test"], test_f_idx)

                new_splits =  {"Train_retain": train_retain, "Train_forget": train_forget, "Test_retain": test_retain, "Test_forget": test_forget}

                self.dataset_splits.update(new_splits)

                if save:
                    self.save_dataset()

            return self.dataset_splits

        elif mode == "random":
            return None
        else:
            return None

