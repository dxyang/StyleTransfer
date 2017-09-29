import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from network import ImageTransformNet
from vgg import Vgg16

# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7
use_cuda = False
dtype = torch.FloatTensor

#---------------------------------------
# Start: For debugging visualization....
#---------------------------------------
dataset_transform_512 = transforms.Compose([
    transforms.Scale(512),                  # scale shortest side to image_size
    transforms.CenterCrop(512),             # crop center image_size out
    transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
    utils.normalize_tensor_transform()      # normalize with ImageNet values
])
dataset_transform_256 = transforms.Compose([
    transforms.Scale(256),                  # scale shortest side to image_size
    transforms.CenterCrop(256),             # crop center image_size out
    transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
    utils.normalize_tensor_transform()      # normalize with ImageNet values
])

testImage_a = utils.load_image("amber.jpg")
testImage_512 = dataset_transform_512(testImage_a)
testImage_256 = dataset_transform_256(testImage_a)
testImage_512 = Variable(testImage_512.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
testImage_256 = Variable(testImage_256.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

testImage2_a = utils.load_image("dan.jpg")
testImage2_512 = dataset_transform_512(testImage2_a)
testImage2_256 = dataset_transform_256(testImage2_a)
testImage2_512 = Variable(testImage2_512.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
testImage2_256 = Variable(testImage2_256.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

testImage3_a = utils.load_image("maine.jpg")
testImage3_512 = dataset_transform_512(testImage3_a)
testImage3_256 = dataset_transform_256(testImage3_a)
testImage3_512 = Variable(testImage3_512.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)
testImage3_256 = Variable(testImage3_256.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

#---------------------------------------
# End: for debugging visualization....
#---------------------------------------

def train(args):            
    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).type(dtype)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.data[0]

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.data[0]

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.data[0]

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 400 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.data[0], content_loss.data[0], tv_loss.data[0]
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0):
                image_transformer.eval()
                outputTestImage_512 = image_transformer(testImage_512).cpu()
                outputTestImage_256 = image_transformer(testImage_256).cpu()
                utils.save_image("Trial7/testImage_1_%d_512_%5d.jpg" %(e, batch_num+1), outputTestImage_512.data[0])
                utils.save_image("Trial7/testImage_1_%d_256_%5d.jpg" %(e, batch_num+1), outputTestImage_256.data[0])

                outputTestImage2_512 = image_transformer(testImage2_512).cpu()
                outputTestImage2_256 = image_transformer(testImage2_256).cpu()
                utils.save_image("Trial7/testImage_2_%d_512_%5d.jpg" %(e, batch_num+1), outputTestImage2_512.data[0])
                utils.save_image("Trial7/testImage_2_%d_256_%5d.jpg" %(e, batch_num+1), outputTestImage2_256.data[0])

                outputTestImage3_512 = image_transformer(testImage3_512).cpu()
                outputTestImage3_256 = image_transformer(testImage3_256).cpu()
                utils.save_image("Trial7/testImage_3_%d_512_%5d.jpg" %(e, batch_num+1), outputTestImage3_512.data[0])
                utils.save_image("Trial7/testImage_3_%d_256_%5d.jpg" %(e, batch_num+1), outputTestImage3_256.data[0])

                print("images saved")
                image_transformer.train()

    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/udnie" + "_" + str(time.ctime()).replace(' ', '_') + "_" + ".model"
    torch.save(image_transformer.state_dict(), filename)
    
    if use_cuda:
        image_transformer.cuda()

def style_transfer(args):
    # content image
    content_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    content = utils.load_image(args.source)
    content = content_transform(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    style_model.load_state_dict(args.model_path)

    # process input image
    stylized = style_model(content)
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print "Current device: %d" %torch.cuda.current_device()

    # command
    if (args.subcommand == "train"):
        print "Training!"
        train(args)
    elif (args.subcommand == "transfer"):
        print "Style transfering!"
        style_transfer(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()








