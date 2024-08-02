import torch
import torchvision
from torchvision import datasets, transforms


def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset == 'cinic-10':
        # cinic_directory = '/path/to/cinic/directory'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

    if dataset == 'tiny_imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

    if dataset == 'imagenet':
        print("Dataset Imagenet, Image size 224")
        imgnet_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        imgnet_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)])

    if dataset == 'imagenet_vgg':
        imgnet_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        imgnet_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)])

        transform_test = transforms.Compose([
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset, train_batch_size, test_batch_size, num_workers=32, root='../data', returnset=False):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    if dataset == 'cinic-10':
        trainset = torchvision.datasets.ImageFolder(root + '/cinic-10/trainval', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/cinic-10/test', transform=transform_test)

    if dataset == 'tiny_imagenet':
        trainset = torchvision.datasets.ImageFolder(root + '/tiny_imagenet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root + '/tiny_imagenet/val', transform=transform_test)

    if dataset == 'imagenet' or dataset == 'imagenet_vgg':
        root = "/rscratch/data/"
        trainset = torchvision.datasets.ImageFolder(root + '/imagenet12/train', transform=transform_train)
        testset  = torchvision.datasets.ImageFolder(root + '/imagenet12/val', transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)
    if returnset:
        return trainset, testset
    return trainloader, testloader


def get_hessianloader(dataset, hessian_batch_size):
    if dataset == 'cifar10':
            hessian_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('../data/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])),
                    batch_size=hessian_batch_size, shuffle=False)
    elif dataset == 'cifar100':
        hessian_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                ])),
                batch_size=hessian_batch_size, shuffle=False)
    elif dataset == 'imagenet':
        transform_train, transform_test = get_transforms(dataset)
        trainset = torchvision.datasets.ImageFolder('/rscratch/data/imagenet12/train', transform=transform_train)
        hessian_loader = torch.utils.data.DataLoader(trainset, batch_size=hessian_batch_size, shuffle=True,
                                              num_workers=32)

    else:
            raise ValueError("No valid dataset is given.")
    return hessian_loader
