import os
import glob
import shutil
import pickle

import torch
import torchvision

from matplotlib import pyplot as plt
from PIL import Image

from constants import VIDEO_BASEPATH, OS_SEPARATOR
from find_smeared_images import predict_image

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Torch Runs on {device}')

    input_folder = VIDEO_BASEPATH + OS_SEPARATOR + 'Raanana_Merkaz' + OS_SEPARATOR + 'cropped_images' + \
                       OS_SEPARATOR + '2022_02_16'
    reverse = True

    model_path = '/home/user1/Dana Porrat/Netivei' + OS_SEPARATOR + 'learn' + OS_SEPARATOR + 'find_smears_model'
    model_filename = model_path + OS_SEPARATOR + 'TorchModel'
    class_names_file = model_path + OS_SEPARATOR + 'class_names.pickle'
    transforms_file = model_path + OS_SEPARATOR + 'transforms.pickle'

    ok_folder = input_folder + OS_SEPARATOR + 'images_ok'
    smeared_folder = input_folder + OS_SEPARATOR + 'images_smeared'
    if not os.path.exists(ok_folder):
        os.mkdir(ok_folder)
    if not os.path.exists(smeared_folder):
        os.mkdir(smeared_folder)

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f'Cannot find model in {model_filename}')
    if not os.path.exists(class_names_file):
        raise FileNotFoundError(f'Cannot find class names in {class_names_file}')
    if not os.path.exists(transforms_file):
        raise FileNotFoundError(f'Cannot find data transforms in {transforms_file}')

    model = torchvision.models.vgg16(pretrained=True)
    num_in = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_in, 2)

    model.load_state_dict(torch.load(model_filename))
    model = model.to(device)
    with open(class_names_file, 'rb') as infile:
        class_names = pickle.load(infile)
    with open(transforms_file, 'rb') as infile:
        data_transforms = pickle.load(infile)

    image_files = glob.glob(input_folder + OS_SEPARATOR + '*.png', reverse=reverse)
    for file in image_files:
        prediction = predict_image(model, file, data_transforms, device)
        prediction = class_names[prediction]
        if 0:
            plt.imshow(Image.open(file))
            plt.title(f'{os.path.basename(file)}: estimation is {prediction}')
            plt.show()
        if prediction == 'ok':
            shutil.move(file, ok_folder)
        elif prediction == 'smeared':
            shutil.move(file, smeared_folder)

