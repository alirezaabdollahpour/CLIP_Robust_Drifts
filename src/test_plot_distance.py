from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector
import torch 
import copy
import json
import argparse
import math
import numpy as np
# from src.args import parse_arguments
import matplotlib.pyplot as plt

# Tangent task vector.
linear_zeroshot_checkpoint = "/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-10/linear_zeroshot.pt" # Pre-trained linearized image encoder.
linear_finetuned_checkpoint = "/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-10/linear_finetuned.pt" # Linearly fine-tuned checkpoint.

linear_task_vector = LinearizedTaskVector(linear_zeroshot_checkpoint, linear_finetuned_checkpoint)


def model_creator(args):
    linear_zeroshot_checkpoint = "/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-10/linear_zeroshot.pt" # Pre-trained linearized image encoder.
    linear_finetuned_checkpoint = "/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-10/linear_finetuned.pt" # Linearly fine-tuned checkpoint.

    linear_task_vector = LinearizedTaskVector(linear_zeroshot_checkpoint, linear_finetuned_checkpoint)

    print(f'linear_task_vector is :{linear_task_vector}')
    θ_zero = linear_task_vector.state_dict()
    print(f'θ_zero is :{θ_zero}')
    
    del θ_zero['classification_head.bias']
    del θ_zero['classification_head.weight']
    # if args.IN == 'ImageNet':
    # print("IN is ImageNet!")
    finetuned_1 = ImageClassifier.load('/home/aabdolla/wise-ft/models/wiseft/ViTB32-ImageNet-IN-ImageNetA-OOD/finetuned/checkpoint_10.pt')
    θ_ImageNet  = finetuned_1.state_dict()
    del θ_ImageNet['classification_head.bias']
    del θ_ImageNet['classification_head.weight']
    # print("Your θ_IN is model on CIFAR10 ")
    
    if args.OOD == "CIFAR10":
        # finetuned_2 = ImageClassifier.load('/home/aabdolla/wise-ft/models/wisef-CIFAR10-train-320epochs-2000sample/finetuned/checkpoint_320.pt')
        finetuned_2 = ImageClassifier.load('/home/aabdolla/wise-ft/models/wisef-CIFAR10-train-320epochs-2000sample/finetuned/checkpoint_320.pt')
        θ_CIFAR10 = finetuned_2.state_dict()
        del θ_CIFAR10['classification_head.bias']    
        del θ_CIFAR10['classification_head.weight']
        return θ_zero,θ_ImageNet,θ_CIFAR10
    elif args.OOD == "CIFAR100":
        finetuned_2 = ImageClassifier.load('/home/aabdolla/wise-ft/train/wise-ft/models/wiseft/ViTB32-CIFAR100/finetuned/checkpoint_10.pt')
        θ_CIFAR10 = finetuned_2.state_dict()
        del θ_CIFAR10['classification_head.bias']    
        del θ_CIFAR10['classification_head.weight']
        return θ_zero,θ_ImageNet,θ_CIFAR10
    
def main(args):
    print('main function running!')
    θ_zero,θ_ImageNet,θ_CIFAR10 = model_creator(args)
    print("Loading checkpoint done!")
    # make sure checkpoints are compatible
    assert set(θ_ImageNet.keys()) == set(θ_zero.keys())
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--OOD",type=str,default = "CIFAR10")
    parser.add_argument("--template",type=str,default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    args = parser.parse_args()
    main(args)
    print('scripts is done!')    