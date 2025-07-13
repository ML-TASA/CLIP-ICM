import argparse
from engine import get_dataset, zero_shot, logistic
import torch
import torch.backends.cudnn as cudnn
from CLIP import clip
from clip_icm import get_A_inv
import csv
import os
from filelock import FileLock


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_results(args, zero_acc, logistic_acc, save_path="results.csv"):
    exclude = {"root", "batch_size", "workers"}

    args_dict = vars(args)
    
    filtered_args = {k: v for k, v in args_dict.items() if k not in exclude}
    
    record = filtered_args.copy()
    record["zero_acc"] = zero_acc
    record["logistic_acc"] = logistic_acc

    file_exists = os.path.isfile(save_path)
    
    lock = FileLock(save_path + ".lock")

    with lock:
        with open(save_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)



def main(args):
    cudnn.benchmark = True
    # load model
    clip_model, preprocess = clip.load(args.arch, device)
    clip_model.eval()
    clip_model = clip_model.float()
    
    # load dataset
    train_loader, _, test_loaders, _, template = get_dataset(args, preprocess=preprocess)
    # train model
    A_inv = get_A_inv(args, clip_model, train_loader)
    # eval zero-shot
    if args.task == 'domain_shift':
        print(f"Target domains: {args.targets}")
        zero_acc = zero_shot(args, clip_model, device, A_inv, test_loaders[0]['loader'], test_loaders[0]['class_names'], template)
        print(f"Zero-shot accuracy: {zero_acc}")
        # eval linear-probe
        logistic_acc = logistic(clip_model, train_loader, test_loaders[0]['loader'], device, A_inv)
        print(f"Linear-probe accuracy: {logistic_acc}")
        save_results(args, zero_acc, logistic_acc)
    else:
        print(f"Target domains: {args.targets}")
        zero_base_acc = zero_shot(args, clip_model, device, A_inv, test_loaders[0]['loader'], test_loaders[0]['class_names'], template)
        zero_open_acc = zero_shot(args, clip_model, device, A_inv, test_loaders[1]['loader'], test_loaders[1]['class_names'], template)
        print(f"Zero-shot base accuracy: {zero_base_acc}")
        print(f"Zero-shot open accuracy: {zero_open_acc}")
        save_results(args, zero_base_acc, zero_open_acc, save_path="results_in_the_wild.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # dataset parameters
    parser.add_argument('--root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS')
    parser.add_argument('--task', default='in_the_wild', choices=
                        ['domain_shift', 'in_the_wild'])
    parser.add_argument('--targets', nargs='+', type=int, default=[0],
                        help='target domain(s) (DomainBed datasets only)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # CLIP-ICM parameters
    
    # ColorJitter
    parser.add_argument('--colorjitter_brightness', type=float)
    parser.add_argument('--colorjitter_contrast', type=float)
    parser.add_argument('--colorjitter_saturation', type=float)
    parser.add_argument('--colorjitter_hue', type=float)
    # Grayscale
    parser.add_argument('--grayscale_num_output_channels', type=int)
    # GaussianBlur
    parser.add_argument('--gaussianblur_kernel_size', type=int)
    parser.add_argument('--gaussianblur_sigma_min', type=float)
    parser.add_argument('--gaussianblur_sigma_max', type=float)
    # RandomInvert
    parser.add_argument('--randominvert_p', type=float)
    # RandomPosterize
    parser.add_argument('--randomposterize_bits', type=int)
    parser.add_argument('--randomposterize_p', type=float)
    # RandomSolarize
    parser.add_argument('--randomsolarize_threshold', type=float)
    parser.add_argument('--randomsolarize_p', type=float)
    # RandomEqualize
    parser.add_argument('--randomequalize_p', type=float)
    # Lambda & Hold Out Dimension
    parser.add_argument('--hold_out_dim', type=int,
                        help='the dimension of content representation to hold out')
    parser.add_argument('--lamda', type=float)
    
    args = parser.parse_args()
    main(args)
