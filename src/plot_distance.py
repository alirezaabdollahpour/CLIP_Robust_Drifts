import json
from collections import OrderedDict
import os
import copy
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




def plot_Alireza(D_IN,ImageNet_ACC_OOD_on_ImageNetA,alphas):
    plt.scatter(D_IN, ImageNet_ACC_OOD_on_ImageNetA,label='D_IN')
    plt.plot(D_IN, ImageNet_ACC_OOD_on_ImageNetA,label='D_IN')
    plt.xlabel('DOOD')
    plt.ylabel('Accuracy on CIFAR10(OOD)')
    plt.title('distance OOD and ACC on CIFAR10')
    
    if not os.path.exists('Linear'):
        os.makedirs('Linear')
    plt.savefig('Linear/DOOD-CIFAR10-ImageNet-IN.png', dpi=300, bbox_inches='tight')

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

def dict_norm_calculator(theta_1):
    u = {key: theta_1[key] for key in theta_1.keys()}
    norm_sum_u = 0.0
    for key in u.keys():
        tensor = u[key]
        norm_squared = torch.norm(tensor) ** 2
        norm_sum_u += norm_squared

    norm_sqrt = torch.sqrt(norm_sum_u)
    return norm_sqrt

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


def Model(args,dataset):
    classification_head = ClassificationHead.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/head_CIFAR10Val.pt')
    image_encoder = LinearizedImageEncoder.load(args.load)
    
    model = ImageClassifier(image_encoder, classification_head)
    
    print("model is created")
    return model,image_encoder,classification_head




def model_creator(args):
    
    linear = LinearizedImageEncoder.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_zeroshot.pt')
    pretrained_checkpoint_ImageNet_zeroshot = ImageClassifier.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_zeroshot.pt')
    finetuned_checkpoint_ImageNet = ImageClassifier.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/ImageNetVal-epochs-10/linear_finetuned.pt')    
    finetuned_checkpoint_CIFAR10 = ImageClassifier.load('/home/aabdolla/tangent_task_arithmetic/checkpoints/ViT-B-32/CIFAR10Val-epochs-15/linear_finetuned.pt')
    del pretrained_checkpoint_ImageNet_zeroshot['model_name']
    del finetuned_checkpoint_ImageNet['model_name']
    del finetuned_checkpoint_CIFAR10['model_name']
    
    return pretrained_checkpoint_ImageNet_zeroshot,finetuned_checkpoint_ImageNet,finetuned_checkpoint_CIFAR10,linear
        
        
        
        
        

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



def main(args):
    print('main function running!')
    θ_zero,θ_ImageNet,θ_CIFAR10,linear = model_creator(args)
    print("Loading checkpoint done!")
    # make sure checkpoints are compatible
    # assert set(θ_ImageNet.keys()) == set(θ_zero.keys())
    # assert set(θ_CIFAR10_1.keys()) == set(θ_zero.keys())
    
    D_IN = []
    D_OOD = []
    D_theta0_theta_OOD = []
    D_theta0_theta_IN = []
    D_theta_OOD_theta_IN = []
    D_theta_alpha_theta_zero = []
    X = []
    split = "test"
    alphas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # alphas = [0,0.1]
    for alpha in alphas:
        theta_alpha, alpha,d_in,d_ood,x,d_theta_alpha_theta_zero = theta_alpha_creator(alpha,θ_ImageNet,θ_CIFAR10,θ_zero) 
        linear.load_state_dict(theta_alpha)
        evaluate(linear, args)
        D_IN.append(d_in)
        D_OOD.append(d_ood)
        X.append(x)
        
        
    ImageNet_ACC_OOD_on_CIFAR10_Linear= ACC_JSON_reader_CIFAR10_IN('/home/aabdolla/wise-ft/results-ImageNet-IN-CIFAR10-OOD.jsonl')
    # print(f'DOOD is :{D_OOD}')
    # print(f'length of DOOD is :{len(D_OOD)}')
    # print(f'length of ACC is :{len(ImageNet_ACC_OOD_on_ImageNetA)}')
    # # ImageNet_ACC_OOD_on_ImageNetA = ACC_JSON_reader_ImageNet_IN('/home/aabdolla/wise-ft/results-ImageNet-OOD-CIFAR10-IN.jsonl')
    # # print(f'len of ACC is :{len(ImageNet_ACC_OOD_on_ImageNetA)}')
    # # plot_Alireza(D_theta_alpha_theta_zero,ImageNet_ACC_OOD_on_ImageNetA,alphas)
    # # plot_Alireza_alpha_distance(alphas,ImageNet_ACC_OOD_on_ImageNetA)
    # # plot_Alireza(alphas,ImageNet_ACC_OOD_on_ImageNetA,alphas) # D_IN,D_OOD,alphas
    # # plot_Alireza_projection_distance(X,ImageNet_ACC_OOD_on_ImageNetA)    
    plot_Alireza(D_OOD,ImageNet_ACC_OOD_on_CIFAR10_Linear,alphas) 
    
    
    # D_OOD = [d.tolist() for d in D_OOD]
    # data_to_save = [{"alpha": alpha, "D_OOD": d_ood} for alpha, d_ood in zip(alphas, D_OOD)]
    # # Save the data to a JSON file
    # with open("DOOD/CIFAR100-DOOD.json", "w") as json_file:
    #     json.dump(data_to_save, json_file)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    print('scripts is done!')    
        