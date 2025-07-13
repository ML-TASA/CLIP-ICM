import time
import copy
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from imagenet_stubs import get_class_names_list
from sklearn.linear_model import LogisticRegression

from CLIP import clip

import converter_domainbed
from utils import accuracy, AverageMeter, ProgressMeter


def get_dataset(args, preprocess):
    if args.task == "domain_shift":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets,
                                                       holdout=0.2, preprocess=preprocess, configs=args)
        train_class_names = class_names
        # train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        train_loader = DataLoader(ConcatDataset(
            train_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers),
                "class_names": class_names
            }
        ]
        template = "a photo of a {}."

    elif args.task == "in_the_wild":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets,
                                                       holdout=0.2, open_ratio=0.5, preprocess=preprocess, configs=args)
        train_class_names = base_class_names
        # train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        train_loader = DataLoader(ConcatDataset(
            train_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."

    return train_loader, val_loader, test_loaders, train_class_names, template


def get_text_features(clip_model, template, class_names, device):
    with torch.no_grad():
        texts = torch.cat(
            [clip.tokenize(template.format(c.replace("_", " ")))
             for c in class_names]).to(device)
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def zero_shot(args, clip_model, device, A_inv, test_loaders, train_class_names, template):

    def get_predictions(model, text_features, device, A_inv):
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loaders)):
                # for images, labels in tqdm(loader):
                image_features = model.encode_image(images.to(device)).float()
                image_features = image_features @ A_inv
                # Calculate the similarity between image features and text features
                similarity = (100.0 * image_features @
                              text_features.T).softmax(dim=-1)
                predictions = similarity.argmax(dim=-1)

                all_predictions.extend(predictions.cpu())
                all_labels.extend(labels)

            # logger.info(f"Progress: {i + 1}/{len(dataset)}")
        return np.array(all_predictions), np.array(all_labels)

    # Calculate the image features and perform evaluation for each domain
    class_labels = train_class_names
    text_features = get_text_features(
        clip_model, template, class_labels, device)
    text_features = text_features @ A_inv
    predictions, labels = get_predictions(clip_model, text_features, device, A_inv)

    # Evaluate using the predictions
    accuracy = np.mean((labels == predictions).astype(float)) * 100.
    print(f"Accuracy on domain {args.targets} = {accuracy:.3f}")
    return accuracy

def logistic(clip_model, train_loader, test_loader, device, A_inv):

    def get_features(loader, model, device):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader)): 
                if type(images) is list:
                    image = images[0]
                    features = model.encode_image(image.to(device))
                else:
                    features = model.encode_image(images.to(device))
                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features), torch.cat(all_labels)

    # Calculate the image features
    train_features, train_labels = get_features(train_loader, clip_model, device)
    train_features = train_features @ A_inv
    test_features, test_labels = get_features(test_loader, clip_model, device)
    test_features = test_features @ A_inv

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)
    # classifier.fit(train_features.float(), train_labels.long())
    # classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, l1_ratio=0.0001,penalty='elasticnet', solver='saga')
    classifier.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())

    # Evaluate using the logistic regression classifier
    # predictions = classifier.predict(test_features.float()).numpy()
    predictions = classifier.predict(test_features.cpu().numpy())

    accuracy = np.mean((test_labels.cpu().numpy() == predictions).astype(float)) * 100.
    return accuracy
