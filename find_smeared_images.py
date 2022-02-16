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


def predict_image(model_, image_file, data_transforms_, device_):
    image = Image.open(image_file).convert('RGB')
    image_tensor = data_transforms_['predict'](image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_ = torch.autograd.Variable(image_tensor)
    input_ = input_.to(device_)
    output = model_(input_)
    index = output.data.cpu().numpy().argmax()
    return index


def predict_smear(model_, image_folder_, class_names_, data_transforms_, device_):
    image_files = glob.glob(image_folder_ + OS_SEPARATOR + '*')
    predictions = []
    plt.figure()
    for file in image_files:
        prediction = predict_image(model_, file, data_transforms_, device_)
        predictions.append(prediction)
        plt.imshow(Image.open(file))
        plt.title(f'{os.path.basename(file)}: estimation is {class_names_[prediction]}')
        plt.show()


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


def run_model_on_data(model_, optimizer_, criterion_, scheduler_, phase):
    if phase == 'train':
        model_.train()  # Set model to training mode
    else:
        model_.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs_, labels in data_loaders[phase]:
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
        scheduler_.step()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    return epoch_loss, epoch_acc


def train_model(model_, criterion_, optimizer_, scheduler_, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0

    training_done = False
    train_loss_prev = 10

    for epoch in range(num_epochs):
        if not training_done:
            print(f'Epoch {epoch}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'validate']:
                epoch_loss, epoch_acc = run_model_on_data(model_, optimizer_, criterion_, scheduler_, phase)
                print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss = epoch_loss
                    # print(f'train loss is {train_loss}, prev loss is {train_loss_prev}, '
                    #       f'loss difference is {np.abs(train_loss - train_loss_prev)}')
                    if np.abs(train_loss - train_loss_prev) < 1e-4:
                        training_done = True
                        print('Training loss converged')
                    else:
                        train_loss_prev = train_loss

                # deep copy the model
                if phase == 'validate' and epoch_acc > best_acc:  # take the first good model, before over-fitting occured
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_.state_dict())
                    print(f'Best accuracy on validate! (so far)')
            print()

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Accuracy: {best_acc:4f}')

    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_


def visualize_model(model_, class_names_, num_images=6):
    print(f'class_names_ is {class_names_}')
    was_training = model_.training
    model_.eval()
    images_so_far = 0
    fig_ = plt.figure()

    with torch.no_grad():
        print(f'class_names_ is {class_names_}')
        for i, (inputs_, labels_) in enumerate(data_loaders['validate']):
            # print(f'i={i}, class_names_ is {class_names_}')
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            # print(f'i={i}, class_names_ is {class_names_}')

            outputs_ = model_(inputs_)
            # print(f'inputs_.size() is {inputs_.size()}')
            # print(f'outputs_.size() is {outputs_.size()}')
            _, predictions_ = torch.max(outputs_, 1)
            # print(f'_ is {_}, predictions_ is {predictions_}')

            # print(f'i={i}, class_names_ is {class_names_}')
            for j in range(inputs_.size()[0]):
                images_so_far += 1
                ax_ = plt.subplot(num_images // 2, 2, images_so_far)
                ax_.axis('off')
                # print(f'i={i}, j={j}, class_names_ is {class_names_}, predictions_ is {predictions_}')
                # print(f'i={i}, j={j}, class_names_ is {class_names_}, predictions_[j] is {predictions_[j]}')
                # print(f'class_names_[predictions_[j]] is {class_names_[predictions_[j]]}')
                # print(f'predictions_[j] is {predictions_[j]}')
                ax_.set_title(f'true: {class_names_[labels_[j]]}, predicted: {class_names_[predictions_[j]]}')
                imshow(inputs_.cpu().data[j])

                if images_so_far == num_images:
                    model_.train(mode=was_training)
                    return
        model_.train(mode=was_training)


def show_results(model_, data_loader_, class_names_, num_images=6):
    print(f'class_names_ is {class_names_}')
    was_training = model_.training
    model_.eval()
    images_so_far = 0
    fig_ = plt.figure()

    with torch.no_grad():
        print(f'class_names_ is {class_names_}')
        for i, (inputs_, labels_) in enumerate(data_loader_['validate']):
            # print(f'i={i}, class_names_ is {class_names_}')
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)
            # print(f'i={i}, class_names_ is {class_names_}')

            outputs_ = model_(inputs_)
            # print(f'inputs_.size() is {inputs_.size()}')
            # print(f'outputs_.size() is {outputs_.size()}')
            _, predictions_ = torch.max(outputs_, 1)
            # print(f'_ is {_}, predictions_ is {predictions_}')

            # print(f'i={i}, class_names_ is {class_names_}')
            for j in range(inputs_.size()[0]):
                images_so_far += 1
                ax_ = plt.subplot(num_images // 2, 2, images_so_far)
                ax_.axis('off')
                # print(f'i={i}, j={j}, class_names_ is {class_names_}, predictions_ is {predictions_}')
                # print(f'i={i}, j={j}, class_names_ is {class_names_}, predictions_[j] is {predictions_[j]}')
                # print(f'class_names_[predictions_[j]] is {class_names_[predictions_[j]]}')
                # print(f'predictions_[j] is {predictions_[j]}')
                ax_.set_title(f'true: {class_names_[labels_[j]]}, predicted: {class_names_[predictions_[j]]}')
                imshow(inputs_.cpu().data[j])

                if images_so_far == num_images:
                    model_.train(mode=was_training)
                    return
        model_.train(mode=was_training)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Torch Runs on {device}')

    image_folder = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                   OS_SEPARATOR + '2022_02_16'

    model_path = '/home/user1/Dana Porrat/Netivei' + OS_SEPARATOR + 'learn' + OS_SEPARATOR + 'find_smears_model'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_filename = model_path + OS_SEPARATOR + 'TorchModel'
    class_names_file = model_path + OS_SEPARATOR + 'class_names.pickle'
    transforms_file = model_path + OS_SEPARATOR + 'transforms.pickle'

    model = torchvision.models.vgg16(pretrained=True)
    # block change in some layers
    if 0:
        for param in model.features.parameters():
            param.requires_grad = False
    num_in = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_in, 2)

    if os.path.exists(model_filename) and os.path.exists(class_names_file) and os.path.exists(transforms_file):
        model.load_state_dict(torch.load(model_filename))
        model = model.to(device)
        with open(class_names_file, 'rb') as infile:
            class_names = pickle.load(infile)
        with open(transforms_file, 'rb') as infile:
            data_transforms = pickle.load(infile)
    else:
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224),
                                              interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                # Resize(224) gives better results
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'validate': torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224),
                                              interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'predict': torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224),
                                              interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                   OS_SEPARATOR + 'find_smears'

        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                                              data_transforms[x])
                          for x in ['train', 'validate']}
        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                       shuffle=True, num_workers=4)
                        for x in ['train', 'validate']}
        class_names = copy.deepcopy(image_datasets['train'].classes)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}

        # Get a batch of training data
        inputs, classes = next(iter(data_loaders['train']))
        print(f'Training inputs.size() is {inputs.size()}')

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        if 0:
            imshow(out, title=[class_names[x] for x in classes])

        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                            num_epochs=51)

        torch.save(model.state_dict(), model_filename)
        print(f'Saved model in {model_filename}')

        with open(class_names_file, 'wb') as outfile:
            pickle.dump(class_names, outfile)
        print(f'Saved class names in {class_names_file}')

        with open(transforms_file, 'wb') as outfile:
            pickle.dump(data_transforms, outfile)
        print(f'Saved data transforms in {transforms_file}')

    # predict
    predict_smear(model, image_folder, class_names, data_transforms, device)

    # evaluate
    model.eval()
    visualize_model(model, class_names)
    plt.show()
