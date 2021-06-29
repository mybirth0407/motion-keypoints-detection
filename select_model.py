import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

def initialize_model(model_name, input_size, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    # custom
    if 'efficientnet' in model_name:
        model_ft = EfficientNet.from_pretrained(
            model_name, num_classes=num_classes
        )
    else:
        model_ft = getattr(models, f'{model_name}')(pretrained=use_pretrained)

    # torchvision
    if "resnet" in model_name:
        """ Resnet
        """
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif "alexnet" in model_name:
        """ Alexnet
        """
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif "vgg" in model_name:
        """ VGG13_bn
        """
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif "squeezenet" in model_name:
        """ Squeezenet
        """
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1,1), stride=(1,1)
        )
        model_ft.num_classes = num_classes

    elif "densenet" in model_name:
        """ Densenet
        """
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif "inception" in model_name:
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    return model_ft, input_size
