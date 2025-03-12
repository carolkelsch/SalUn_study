#import copy
import os
from collections import OrderedDict
import sys
#import arg_parser
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import random
import numpy as np
from dataset_split import *

def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_gradient_ratio(data_loaders, model, criterion, **kwargs):
    optimizer = torch.optim.SGD(
        model.parameters(),
        kwargs['unlearn_lr'],
        momentum=kwargs['momentum'],
        weight_decay=kwargs['weight_decay'],
    )

    gradients = {}

    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(kwargs['save_dir'], "with_{}.pt".format(i)))


def generate_fmask(model, batch_size: int, **kwargs):
    #args = arg_parser.parse_args()
    print('entered function...')

    if torch.cuda.is_available():
        torch.cuda.set_device(int(0))
        device = torch.device(f"cuda:{int(0)}")
    else:
        device = torch.device("cpu")

    os.makedirs(kwargs['save_dir'], exist_ok=True)

    model.cuda()
    
    '''
    if args.seed:
        setup_seed(args.seed)
        seed = args.seed
    else:
    '''
    setup_seed(1)

    current_path = os.getcwd()
    root_dataset = current_path + '/datasets/' ## TODO: should not be hardcoded
    ds = UnlearnDatasetSplit(root_dataset, "cifar10")
    
    print('loaded dataset...')

    splitted = ds.split_dataset("class", False, forget=['cat','frog'])
    print(list(splitted.keys()))

    sets = np.array(['Train','Test','Train_retain','Test_retain','Train_forget','Test_forget'])
    print(np.isin(sets, np.array(list(splitted.keys()))).sum())
    if np.isin(sets, np.array(list(splitted.keys()))).sum() != 6:
        print(f"\033[31mNeeded datasets are not available\r\n Please pass the dictionary with the following datasets{sets}!\033[0m")
        return None
    
    print('found all dataset splits...')
    
    retain_loader = ds.get_loader('Train_retain',
        batchsize=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )
    forget_loader = ds.get_loader('Train_forget',
        batchsize=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )
    
    
    print('created loaders...')

    assert len(splitted['Train_forget']) + len(splitted['Train_retain']) == len(splitted['Train']), print(f'Sizes dont match {len(splitted["Train_retain"])} {len(splitted["Train_forget"])} {len(splitted["Train"])}')

    print(f"number of retain dataset {len(splitted['Train_retain'])}")
    print(f"number of forget dataset {len(splitted['Train_forget'])}")

    # Validation loader is only 10 percent of the train dataset with no further
    # changes according to the forget or retain set, and is not used here

    unlearn_data_loaders = OrderedDict(forget=forget_loader)

    criterion = nn.CrossEntropyLoss()
    '''
    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
    
    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    
    else:
    '''
    checkpoint = torch.load(kwargs['model_path'], map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
    
    print('running unlearning...')

    save_gradient_ratio(unlearn_data_loaders, model, criterion, kwargs)