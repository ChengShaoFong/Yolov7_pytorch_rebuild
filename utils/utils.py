import os
import random
import time
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def __image_size(dataset):
    # un-squeeze is used here to add the batch dimension (value=1), which is missing
    return dataset[0][0].unsqueeze(0).size()

def prepare_dataLoader(datasetName, img_size=(32, 32), num_workers=4, train_batch_size=128, eval_batch_size=100):
    # For CIFAR10 dataset
    if datasetName == "cifar10":  
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size[0], padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root="./dataset/cifar10", train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(
            root="./dataset/cifar10", train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = len(classes)

    # For CIFAR100 dataset
    if datasetName == "cifar100":
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size[0], padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root="./dataset/cifar100", train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR100(
            root="./dataset/cifar100", train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

        num_classes = 100
    
    # For MNIST dataset
    elif datasetName == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), # 這樣 normalize 去修正輸入影像，不然讀取會錯
        ])

        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), # 這樣 normalize 去修正輸入影像，不然讀取會錯
        ])

        trainset = torchvision.datasets.MNIST(
            root="./dataset/mnist", train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.MNIST(
            root="./dataset/mnist", train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

        classes = ('zero', 'one', 'two', 'three', 'four',
                   'five', 'six', 'seven', 'eight', 'nine')
        num_classes = len(classes)
    
    # For ImageNet dataset
    elif datasetName == "imagenet-tiny":
        img_size =(224, 224)
        train_dir = os.path.join("./dataset/tiny-imagenet-200", 'train')
        test_dir = os.path.join("./dataset/tiny-imagenet-200", 'val') # ori : val

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(img_size),
                # transforms.RandomCrop(img_size[0], padding=0),  #if 32*32
                transforms.RandomResizedCrop(img_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
        test_dataset = datasets.ImageFolder(
            test_dir, 
            transforms.Compose([
                transforms.Resize(img_size), # ori : 256
                transforms.CenterCrop(img_size[0]), #中心裁剪
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
        num_train = len(train_dataset)
        indices = list(range(num_train))

        valid_size = 0.0
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        input_shape = __image_size(train_dataset)

        
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=50, shuffle=True, num_workers=num_workers, pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=50, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        num_classes = 200

    return trainloader, testloader, num_classes

def measure_inference_latency(model, device, input_size=(1, 3, 32, 32), num_samples=100, num_warmups=10):
    model.to(device)
    model.eval()
    
    x = torch.rand(size=input_size).to(device)
    
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples
    
    return elapsed_time, elapsed_time_ave