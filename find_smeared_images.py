# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
import glob
import pickle
import time
import copy

import numpy as np
import torch
import torchvision

from PIL import Image
import cv2

import math
from matplotlib import pyplot as plt

from constants import VIDEO_BASEPATH, OS_SEPARATOR


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model_, criterion_, optimizer_, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                model_.train()  # Set model to training mode
            else:
                model_.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs_, labels in dataloaders[phase]:
                inputs_ = inputs_.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_(inputs_)
                    _, predictions_ = torch.max(outputs, 1)
                    loss_ = criterion_(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_.backward()
                        optimizer_.step()

                # statistics
                running_loss += loss_.item() * inputs_.size(0)
                running_corrects += torch.sum(predictions_ == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_


def visualize_model(model_, num_images=6):
    was_training = model_.training
    model_.eval()
    images_so_far = 0
    fig_ = plt.figure()

    with torch.no_grad():
        for i, (inputs_, labels) in enumerate(dataloaders['val']):
            inputs_ = inputs_.to(device)
            labels = labels.to(device)

            outputs = model_(inputs_)
            _, predictions_ = torch.max(outputs, 1)

            for j in range(inputs_.size()[0]):
                images_so_far += 1
                ax_ = plt.subplot(num_images//2, 2, images_so_far)
                ax_.axis('off')
                ax_.set_title('predicted: {}'.format(class_names[predictions_[j]]))
                imshow(inputs_.cpu().data[j])

                if images_so_far == num_images:
                    model_.train(mode=was_training)
                    return
        model_.train(mode=was_training)


data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
           OS_SEPARATOR + 'find_smears'
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                                      data_transforms[x])
                  for x in ['train', 'validate']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'validate']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

if 0:
    imshow(out, title=[class_names[x] for x in classes])

if 0:
    data_dict_file = r'/home/user1/Dana Porrat/Netivei/learn/learn_smear_training_data.pkl'
    if os.path.exists(data_dict_file):
        data_dict = pickle.load(open(data_dict_file, "rb"))
        print(f'Loaded file {data_dict_file}')
        x_list = data_dict['x_list']
        y_list = data_dict['y_list']
        x_mean = data_dict['x_mean']
        x_std = data_dict['x_std']
    else:
        image_ok_folder = r'/home/user1/Dana Porrat/Netivei/videos/Raanana_Merkaz/cropped_images/images_ok'
        image_smeared_folder = '/home/user1/Dana Porrat/Netivei/videos/Raanana_Merkaz/cropped_images/images_smeared'

        image_ok_files = glob.glob(image_ok_folder + '/*.png')
        image_smeared_files = glob.glob(image_smeared_folder + '/*.png')

        x_list = []
        y_list = []
        x_mean = []
        x_std = []

        for ii_file, file in enumerate(image_ok_files + image_smeared_files):
            if ii_file % 100 == 0:
                print(f'File {ii_file} of {len(image_ok_files + image_smeared_files)} ...')
            img = Image.open(file).convert('RGB')
            img_np = np.array(img)
            if 1:
                img_224 = cv2.resize(img_np, (224, 224))
            else:
                shape = img_np.shape
                y_start = (shape[0] - 224) // 2
                x_start = (shape[1] - 224) // 2
                img_224 = img_np[y_start:y_start + 224, x_start:x_start + 224, :]
            if 0:
                fig, ax = plt.subplots(2, 1)
                ax[0].imshow(img_np)
                ax[1].imshow(img_224)
                plt.show()

            # img_224_colorfirst = np.moveaxis(img_224, -1, 0)  # channel RGB, height, width
            x_list.append(img_224)
            x_mean.append(np.mean(img_224, axis=(0, 1)))
            x_std.append(np.std(img_224, axis=(0, 1)))
            if file in image_ok_files:
                y_list.append(1)
            elif file in image_smeared_files:
                y_list.append(0)
            else:
                raise ValueError(f'Problem with file {file}')

        data_dict = {'x_list': x_list, 'y_list': y_list, 'x_mean': x_mean, 'x_std': x_std}
        pickle.dump(data_dict, open(data_dict_file, "wb"))
        print(f'Saved file {data_dict_file}')

    img_mean = np.mean(np.array(x_mean), axis=0)
    img_std = np.mean(np.array(x_std), axis=0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    # toPILImage = transforms.ToPILImage(mode='RGB')

    img_224_normalized = []
    for image in x_list:
        img_224_normalized.append(normalize(toTensor(image)))

        # un-normalize image and show
        if 0:
            img_unnorm = img_224_normalized[-1].cpu().detach().numpy().transpose((1, 2, 0))
            for ii in range(3):
                img_unnorm[:, :, ii] = img_std[ii] * img_unnorm[:, :, ii] + img_mean[ii]
            img_unnorm -= np.min(img_unnorm)
            img_unnorm = img_unnorm.astype(np.uint8)

            if 0:
                fig, ax = plt.subplots(2, 2)
                ax[0][0].imshow(image)
                ax[0][1].imshow(img_unnorm[:, :, 0], cmap='gray')
                ax[0][1].set_title('0-R')

                ax[1][0].imshow(img_unnorm[:, :, 1], cmap='gray')
                ax[1][0].set_title('1-G')

                ax[1][1].imshow(img_unnorm[:, :, 2], cmap='gray')
                ax[1][1].set_title('2-B')

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)
            ax[1].imshow(img_unnorm)

            plt.show()

model = torchvision.models.vgg16(pretrained=True)
model.classifier[6].out_features = 2

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                    num_epochs=25)
