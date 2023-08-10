import json
from collections import OrderedDict
import os
import copy
import os
import time
import torch.nn.functional as F
import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import LabelSmoothing, cosine_lr
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.heads import *
from src.eval import *

import matplotlib.pyplot as plt
import torch
# Function definition to check validity
def is_valid_triangle(a,b,c):
    if a+b>=c and b+c>=a and c+a>=b:
        return True
    else:
        return False




def plot_Alireza(D_IN,ImageNet_ACC_OOD_on_ImageNetA,alphas,args):
    plt.scatter(alphas, ImageNet_ACC_OOD_on_ImageNetA,label='D_IN')
    plt.plot(alphas, ImageNet_ACC_OOD_on_ImageNetA,label='D_IN')
    plt.xlabel('alphas')
    if args.OOD == "CIFAR10":
        
        plt.ylabel('Accuracy on CIFAR10(OOD)')
        plt.title('alphas and ACC on CIFAR10')
        if not os.path.exists('Linear'):
            os.makedirs('Linear')
        plt.savefig('Linear/alphas-CIFAR10-ImageNet-IN.png', dpi=300, bbox_inches='tight')
        
    if args.OOD == "ImageNet":        
        plt.ylabel('Accuracy on ImageNet(OOD)')
        plt.title('distance OOD and ACC on ImageNet')
        if not os.path.exists('Linear'):
            os.makedirs('Linear')
        plt.savefig('Linear/DOOD-CIFAR10-ImageNet-OOD.png', dpi=300, bbox_inches='tight')

def plot_Alireza_alpha_distance(alphas,D_IN):
    
    # plt.scatter(alphas, D_OOD,label='D_OOD')
    # plt.plot(alphas, D_OOD, label='D_OOD')
    plt.scatter(alphas,D_IN,label='D_IN')
    plt.plot(alphas,D_IN,label='D_IN')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy on OOD(ImageNet) for moel FT on CIFAR10')
    # plt.xlabel('alpha')
    # plt.ylabel('D_OOD')
    plt.title('ALPHA and Accuracy')
    # plt.legend()
    plt.savefig('ALPHA-Accuracy-ImageNet-OOD-CIFAR10-IN.png', dpi=300, bbox_inches='tight')
    
def plot_Alireza_projection_distance(x,ACC):
    
    # plt.scatter(alphas, D_OOD,label='D_OOD')
    # plt.plot(alphas, D_OOD, label='D_OOD')
    plt.scatter(x,ACC,label='D_IN')
    plt.plot(x,ACC,label='D_IN')
    plt.xlabel('distance of P(theta_alpha) and CIFAR10')
    plt.ylabel('Accuracy on OOD(CIFAR10)')
    # plt.xlabel('alpha')
    # plt.ylabel('D_OOD')
    plt.title('distance and Accuracy')
    # plt.legend()
    plt.savefig('distance-2-Accuracy-ImageNet-IN-CIFAR10-OOD-v.png', dpi=300, bbox_inches='tight')
    
def plot_Alireza_cosine_distance(x,ACC):
    
    # plt.scatter(alphas, D_OOD,label='D_OOD')
    # plt.plot(alphas, D_OOD, label='D_OOD')
    plt.scatter(x,ACC,label='D_IN')
    plt.plot(x,ACC,label='D_IN')
    plt.xlabel('cosine of beta')
    plt.ylabel('Accuracy on OOD(CIFAR10)')
    # plt.xlabel('alpha')
    # plt.ylabel('D_OOD')
    plt.title('cosine and Accuracy')
    # plt.legend()
    plt.savefig('cosine-Accuracy-ImageNet-IN-CIFAR10-OOD-last.png', dpi=300, bbox_inches='tight')
    
def ACC_JSON_reader_ImageNet_IN(path):
    ImageNet_top1_acc = []

    with open(path) as file:
        for line in file:
            try:
                data = json.loads(line)
                # if "ImageNet:top1" in data:
                if "current_epoch" in data and data["current_epoch"] == 9 and "ImageNet:top1" in data:
                    print("come here")
                    ImageNet_top1_acc.append(data["ImageNet:top1"])
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    # Print the list of CIFAR10:top1 values
    print(ImageNet_top1_acc)
    return ImageNet_top1_acc

def ACC_JSON_reader_CIFAR101_IN(path):
    cifar10_top1_acc = []

    with open(path) as file:
        for line in file:
            try:
                data = json.loads(line)
                if "CIFAR101:top1" in data:
                    cifar10_top1_acc.append(data["CIFAR101:top1"])
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    # Print the list of CIFAR10:top1 values
    print(cifar10_top1_acc)
    return cifar10_top1_acc

def ACC_JSON_reader_CIFAR10_IN(path):
    cifar10_top1_acc = []

    with open(path) as file:
        for line in file:
            try:
                data = json.loads(line)
                if "CIFAR10:top1" in data:
                    cifar10_top1_acc.append(data["CIFAR10:top1"])
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    # Print the list of CIFAR10:top1 values
    print(cifar10_top1_acc)
    return cifar10_top1_acc

def ACC_JSON_reader_CIFAR100_IN(path):
    cifar100_top1_acc = []

    with open(path) as file:
        for line in file:
            try:
                data = json.loads(line)
                if "CIFAR100:top1" in data:
                    cifar100_top1_acc.append(data["CIFAR100:top1"])
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    # Print the list of CIFAR10:top1 values
    print(cifar100_top1_acc)
    return cifar100_top1_acc
def dict_to_orderdict(dict_data):
    converted_regular_dict = OrderedDict([(key, value) for key, value in dict_data.items()])
    return converted_regular_dict

def multiply_dict_values_by_constant(input_dict, gamma):
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[key] = value * gamma
    return output_dict



def dict_distance(theta_1,theta_2):
    u = {key: theta_2[key]-theta_1[key] for key in theta_1.keys()}
    norm_sum_u = 0.0
    for key in u.keys():
        tensor = u[key]
        norm_squared = torch.norm(tensor) ** 2
        norm_sum_u += norm_squared

    norm_sqrt = torch.sqrt(norm_sum_u)
    return norm_sqrt

def dict_distance_2(model_1, model_2):
    distance = 0.0
    count = 0

    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        diff = param_2 - param_1
        distance += torch.norm(diff) ** 2
        # count += diff.numel()  

    distance = torch.sqrt(distance)
    return distance




def dict_norm_calculator(theta_1):
    u = {key: theta_1[key] for key in theta_1.keys()}
    norm_sum_u = 0.0
    for key in u.keys():
        tensor = u[key]
        norm_squared = torch.norm(tensor) ** 2
        norm_sum_u += norm_squared

    norm_sqrt = torch.sqrt(norm_sum_u)
    return norm_sqrt

def dict_norm_calculator_2(model_1, model_2):
    distance = 0.0
    count = 0

    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        diff = param_2 - param_1
        distance += torch.norm(diff) ** 2
        count += diff.numel()  

    distance = torch.sqrt(distance / count)
    return distance

# theta_zero,theta_alpha,theta_OOD,mode="x"
def dot_product_difference(theta_zero,theta_alpha,theta_CIFAR,mode): #theta_zero,theta_IN,theta_OOD,mode="x"
    if mode == "x":
        print("calculate x")
        u = {key: theta_CIFAR[key] - theta_zero[key] for key in theta_zero.keys()}
        v = {key: theta_alpha[key] - theta_zero[key] for key in theta_zero.keys()}

        dot_product = 0.0
        for key in u.keys():
            tensor_u = u[key]
            tensor_v = v[key]
            dot_product += torch.sum(tensor_u * tensor_v)

        # x = dot_product/dict_distance(theta_CIFAR,theta_zero)/dict_distance(theta_alpha, theta_zero) # for cosine calc
        x = dot_product/dict_distance(theta_CIFAR,theta_zero)
    else:
        print("calculate y")
        u = {key: theta_zero[key] - theta_CIFAR[key] for key in theta_zero.keys()}
        v = {key: theta_alpha[key] - theta_CIFAR[key] for key in theta_alpha.keys()}

        dot_product = 0.0
        for key in u.keys():
            tensor_u = u[key]
            tensor_v = v[key]
            dot_product += torch.sum(tensor_u * tensor_v)

        x = dot_product/dict_distance(theta_CIFAR,theta_zero)
        
    return x

# theta_zero,theta_alpha,theta_OOD,mode="x"
def dot_product_difference_2(theta_zero,theta_alpha,theta_CIFAR,theta_ImageNet,mode): #theta_zero,theta_IN,theta_OOD,mode="x"
    if mode == "u":
        print("calculate u")
        u = {key: theta_CIFAR[key] - theta_alpha[key] for key in theta_alpha.keys()}
        v = {key: theta_CIFAR[key] - theta_ImageNet[key] for key in theta_ImageNet.keys()}

        dot_product = 0.0
        for key in u.keys():
            tensor_u = u[key]
            tensor_v = v[key]
            dot_product += torch.sum(tensor_u * tensor_v)

        # x = dot_product/dict_distance(theta_CIFAR,theta_zero)/dict_distance(theta_alpha, theta_zero) # for cosine calc
        x = dot_product/dict_distance(theta_CIFAR,theta_ImageNet)
    else:
        print("calculate v")
        u = {key: theta_ImageNet[key] - theta_alpha[key] for key in theta_alpha.keys()}
        v = {key: theta_ImageNet[key] - theta_CIFAR[key] for key in theta_alpha.keys()}

        dot_product = 0.0
        for key in u.keys():
            tensor_u = u[key]
            tensor_v = v[key]
            dot_product += torch.sum(tensor_u * tensor_v)

        x = dot_product/dict_distance(theta_CIFAR,theta_ImageNet)
        
    return x

def dot_product_cosine_projction(theta_zero,theta_alpha,theta_ImageNet): # theta_OOD,theta_alpha,theta_IN
    # theta_zero is theta_OOD
    # u = {key: theta_zero[key] - theta_alpha[key] for key in theta_zero.keys()}
    # v = {key: theta_alpha[key] - theta_ImageNet[key]  for key in theta_alpha.keys()}
    u = {key: theta_alpha[key] - 0 for key in theta_alpha.keys()}
    v = {key: theta_zero[key] - 0  for key in theta_zero.keys()}
    norm_u = dict_norm_calculator(u)
    norm_v = dict_norm_calculator(v)
    dot_product = 0.0 
    for key in u.keys():
        tensor_u = u[key]
        tensor_v = v[key]
        dot_product += torch.sum(tensor_u * tensor_v)
        
    x = (dot_product)/((norm_u)*(norm_v))
    
    return x

def cosine_between_theta_alpha_theta_OOD(theta_zero,theta_alpha,theta_ImageNet,theta_OOD): # theta_OOD,theta_alpha,theta_IN,theta_OOD
    # theta_zero is theta_OOD
    # u = {key: theta_zero[key] - theta_alpha[key] for key in theta_zero.keys()}
    # v = {key: theta_alpha[key] - theta_ImageNet[key]  for key in theta_alpha.keys()}
    u = {key: theta_alpha[key] - theta_OOD[key]  for key in theta_alpha.keys()}
    v = {key: theta_alpha[key] - theta_zero[key]  for key in theta_zero.keys()}
    norm_u = dict_norm_calculator(u)
    norm_v = dict_norm_calculator(v)
    dot_product = 0.0
    for key in u.keys():
        tensor_u = u[key]
        tensor_v = v[key]
        dot_product += torch.sum(tensor_u * tensor_v)
        
    x = (dot_product)/((norm_u)*(norm_v))
    
    return x







def model_creator(args):
    
    if args.OOD == "CIFAR10":
        print("OOD IS CIFAR10")
        pretrained_checkpoint_ImageNet = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_zeroshot.pt'
        finetuned_checkpoint_ImageNet = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_finetuned.pt'
        task_vector_zeroshot = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_ImageNet)
        zs_encoder_ImageNet = task_vector_zeroshot.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        ft_encoder_ImageNet = task_vector_zeroshot.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        linear_image_encoder_ImageNet_zeroshot = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_ImageNet, args=args)
        
    
        task_vector = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_ImageNet)
        zs_encoder_ImageNet = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        ft_encoder_ImageNet = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=1.0)
        linear_image_encoder_ImageNet = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_ImageNet, args=args)
        
        finetuned_checkpoint_CIFAR10 = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_finetuned.pt'
        task_vector = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_CIFAR10)
        ft_encoder_CIFAR10 = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=1.0)
        linear_image_encoder_CIFAR10 = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_CIFAR10, args=args)
        
        distance_zs_OOD = dict_distance_2(linear_image_encoder_ImageNet_zeroshot,linear_image_encoder_CIFAR10)
        distance_IN_OOD = dict_distance_2(linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10)
        distance_zs_IN = dict_distance_2(linear_image_encoder_ImageNet_zeroshot,linear_image_encoder_ImageNet)
        
        # print("*"*80)
        # print(f'distance_zs_OOD is :{distance_zs_OOD}')
        # print("*"*80)
        # print(f'distance_IN_OOD is :{distance_IN_OOD}')
        # print("*"*80)
        # print(f'distance_zs_IN is :{distance_zs_IN}')
        # print("*"*80)
        
        return linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10,linear_image_encoder_ImageNet_zeroshot
    
    elif args.OOD == "CIFAR100":
        print("OOD IS CIFAR100")
        pretrained_checkpoint_ImageNet = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_zeroshot.pt'
        finetuned_checkpoint_ImageNet = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_finetuned.pt'
        task_vector_zeroshot = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_ImageNet)
        zs_encoder_ImageNet = task_vector_zeroshot.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        ft_encoder_ImageNet = task_vector_zeroshot.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        linear_image_encoder_ImageNet_zeroshot = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_ImageNet, args=args)
        
    
        task_vector = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_ImageNet)
        zs_encoder_ImageNet = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        ft_encoder_ImageNet = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=1.0)
        linear_image_encoder_ImageNet = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_ImageNet, args=args)
        
        finetuned_checkpoint_CIFAR10 = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_finetuned.pt'
        task_vector = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_CIFAR10)
        ft_encoder_CIFAR10 = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=1.0)
        linear_image_encoder_CIFAR10 = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_CIFAR10, args=args)
        
        # distance_zs_OOD = dict_distance_2(linear_image_encoder_ImageNet_zeroshot,linear_image_encoder_CIFAR10)
        # distance_IN_OOD = dict_distance_2(linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10)
        # distance_zs_IN = dict_distance_2(linear_image_encoder_ImageNet_zeroshot,linear_image_encoder_ImageNet)
        
        # print("*"*80)
        # print(f'distance_zs_OOD is :{distance_zs_OOD}')
        # print("*"*80)
        # print(f'distance_IN_OOD is :{distance_IN_OOD}')
        # print("*"*80)
        # print(f'distance_zs_IN is :{distance_zs_IN}')
        # print("*"*80)
        
        return linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10,linear_image_encoder_ImageNet_zeroshot
    
    elif args.OOD =="ImageNet":
        print("ImageNet")
        pretrained_checkpoint_ImageNet = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_zeroshot.pt'
        finetuned_checkpoint_ImageNet = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_finetuned.pt'
        task_vector_zeroshot = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_ImageNet)
        zs_encoder_ImageNet = task_vector_zeroshot.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        ft_encoder_ImageNet = task_vector_zeroshot.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        linear_image_encoder_ImageNet_zeroshot = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_ImageNet, args=args)
        
    
        task_vector = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_ImageNet)
        zs_encoder_ImageNet = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=0.0)
        ft_encoder_ImageNet = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=1.0)
        linear_image_encoder_ImageNet = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_ImageNet, args=args)
        
        finetuned_checkpoint_CIFAR10 = '/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_finetuned.pt'
        task_vector = LinearizedTaskVector(pretrained_checkpoint_ImageNet, finetuned_checkpoint_CIFAR10)
        ft_encoder_CIFAR10 = task_vector.apply_to(pretrained_checkpoint_ImageNet, scaling_coef=1.0)
        linear_image_encoder_CIFAR10 = LinearizedImageEncoder(
            init_encoder=zs_encoder_ImageNet, image_encoder=ft_encoder_CIFAR10, args=args)
        
        # distance_zs_OOD = dict_distance_2(linear_image_encoder_ImageNet_zeroshot,linear_image_encoder_CIFAR10)
        # distance_IN_OOD = dict_distance_2(linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10)
        # distance_zs_IN = dict_distance_2(linear_image_encoder_ImageNet_zeroshot,linear_image_encoder_ImageNet)
        
        # print("*"*80)
        # print(f'distance_zs_OOD is :{distance_zs_OOD}')
        # print("*"*80)
        # print(f'distance_IN_OOD is :{distance_IN_OOD}')
        # print("*"*80)
        # print(f'distance_zs_IN is :{distance_zs_IN}')
        # print("*"*80)
        
        return linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10,linear_image_encoder_ImageNet_zeroshot
    
    # elif args.OOD == "ImageNet":
    #     linear = LinearizedImageEncoder.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_zeroshot.pt')
    #     pretrained_checkpoint_CIFAR10_zeroshot = ImageClassifier.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_zeroshot.pt')
    #     finetuned_checkpoint_ImageNet = ImageClassifier.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_finetuned.pt')    
    #     finetuned_checkpoint_CIFAR10 = ImageClassifier.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_finetuned.pt')
    #     del pretrained_checkpoint_ImageNet_zeroshot['model_name']
    #     del finetuned_checkpoint_ImageNet['model_name']
    #     del finetuned_checkpoint_CIFAR10['model_name']
    #     return pretrained_checkpoint_CIFAR10_zeroshot,finetuned_checkpoint_ImageNet,finetuned_checkpoint_CIFAR10,
        
        
        
        
        
        

def theta_alpha_creator(alpha, theta_IN, theta_OOD, theta_zero): # alpha,θ_ImageNet,θ_CIFAR10,θ_zero
    alpha = alpha
    theta_alpha = {key: alpha * theta_IN[key] + (1-alpha) * theta_zero[key] for key in theta_IN.keys()}
    
    D_theta0_theta_OOD = dict_distance(theta_zero,theta_OOD)
    D_theta0_theta_IN = dict_distance(theta_zero,theta_IN)
    D_theta_OOD_theta_IN = dict_distance(theta_OOD,theta_IN)
    D_OOD = dict_distance(theta_alpha,theta_OOD)
    D_IN = dict_distance(theta_alpha,theta_IN)
    D_theta_alpha_theta_zero = dict_distance(theta_alpha,theta_zero)
    
    # x = dot_product_difference(theta_zero,theta_alpha,theta_OOD,mode="y") # theta_zero,theta_alpha,theta_CIFAR,mode
    # x = dot_product_cosine_projction(theta_OOD,theta_alpha,theta_IN) # theta_zero,theta_alpha,theta_ImageNet
    # x = dot_product_difference(theta_zero,theta_alpha,theta_OOD,mode="y") # theta_zero,theta_alpha,theta_CIFAR,mode
    # x = dot_product_difference_2(theta_zero,theta_alpha,theta_OOD,theta_IN,mode='v')
    x = cosine_between_theta_alpha_theta_OOD(theta_zero,theta_alpha,theta_IN,theta_OOD)
    print(f'results for alpha :{alpha}')
    print(f'D_theta0_theta_OOD is :{D_theta0_theta_OOD}')
    print(f'D_theta0_theta_IN is : {D_theta0_theta_IN}')
    print(f'D_theta_OOD_theta_IN is :{D_theta_OOD_theta_IN}')
    return theta_alpha, alpha,D_IN,D_OOD,x,D_theta_alpha_theta_zero

def gram_schmidt(gradients, u):
    for gradient in gradients:
        dot_product = torch.dot(u.flatten(), gradient.flatten())
        norm_squared = torch.dot(gradient.flatten(), gradient.flatten())
        projection = (dot_product / norm_squared) * gradient
        u -= projection
    return u

def finetune(image_encoder,args):

    train_dataset = args.train_dataset
    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    # model.freeze_head()
    model = model.cuda()
    print(f'batch-size is :{args.batch_size}')
    preprocess_fn = model.train_preprocess
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    print(f'dataset is :{dataset}')
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)
    
    

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    model.eval()
    counter = 0
    Grad = []
    for i, batch in enumerate(data_loader):
        if counter == 1:
            break  

        batch = maybe_dictionarize(batch)
        inputs = batch["images"].cuda()
        labels = batch["labels"].cuda()
        
        logits = model(inputs)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        gradients = []
        for param in model.parameters():
            # print("calculate gradients")
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        # optimizer.step()
        # print(f'gradients are in loop is :{gradients}')
        Grad.append(gradients)
        
        counter += 1  
    return Grad



def main(args):
    torch.cuda.empty_cache()
    linear_image_encoder_ImageNet,linear_image_encoder_CIFAR10,linear_image_encoder_ImageNet_zeroshot = model_creator(args)
    train_datasets = [
        "CIFAR10",# "CIFAR100",# "ImageNet",
    ]
    epochs = {
        # "CIFAR10" : 15,
        "CIFAR10": 1,
        # "ImageNet":10,
    }

    for dataset in train_datasets:
        args = parse_arguments()
        args.lr = 1e-5
        args.num_grad_accumulation = 1
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        gradients = finetune(linear_image_encoder_CIFAR10, args)
    
    print(f'gradients is :{gradients}')
    # Convert gradients to a tensor and normalize
    gradients_tensor = torch.stack([torch.cat(grad) for grad in gradients])
    norms = torch.norm(gradients_tensor, dim=1)
    normalized_gradients = gradients_tensor / norms[:, None]
    
    print("="*80)
    print(f'normalized_gradients shape is :{normalized_gradients.shape}')
    print("="*80)
    gradient_dim = normalized_gradients.shape
    random_vector = torch.randn(gradient_dim)
    # Normalize the random vector
    normalized_random_vector = F.normalize(random_vector, p=2, dim=0)
    
    # Perform Gram-Schmidt orthogonalization
    orthogonalized_u = gram_schmidt(normalized_gradients, normalized_random_vector.cuda())
    
    # Verify orthogonality
    dot_products = [torch.dot(orthogonalized_u.flatten(), grad.flatten()) for grad in normalized_gradients]
    
    print("Dot products between orthogonalized u and gradients:")
    for i, dot_product in enumerate(dot_products):
        print(f"Sample {i+1}: {dot_product.item()}")

    print("Final orthogonalized u:", orthogonalized_u)
    


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    print('scripts is done!')    
        
