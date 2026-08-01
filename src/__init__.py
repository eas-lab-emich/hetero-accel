"""Hard-coded paths to relevant tools and files
"""
import os

project_dir = os.path.dirname(os.path.dirname(__file__))

dataset_dirs = {
        'mnist': os.path.join(os.path.dirname(project_dir), 'data', 'mnist'),
        'cifar10': os.path.join(os.path.dirname(project_dir), 'data', 'cifar10'),
        'cifar100': os.path.join(os.path.dirname(project_dir), 'data', 'cifar100'),
        'imagenet': os.path.join(os.path.dirname(project_dir), 'data', 'Imagenet'),
        'voc_det': os.path.join(os.path.dirname(project_dir), 'data', 'vocdet'),
        'voc_seg': os.path.join(os.path.dirname(project_dir), 'data', 'vocseg'),
        'coco': os.path.join(os.path.dirname(project_dir), 'data', 'mscoco'),
        'multi30k': os.path.join(os.path.dirname(project_dir), 'data', 'multi30k'),
        }

cifar10_pretrained_paths = {
        'vgg11_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'vgg11_cifar10.pth'),
        'vgg13_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'vgg13_cifar10.pth'),
        'vgg16_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'vgg16_cifar10.pth'),
        'vgg19_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'vgg19_cifar10.pth'),
        'resnet18_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'resnet18_cifar10.pth'),
        'resnet34_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'resnet34_cifar10.pth'),
        'resnet50_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'resnet50_cifar10.pth'),
        'mobilenet_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'mobilenet_cifar10.pth'),
        'mobilenetv2_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'mobilenetv2_cifar10.pth'),
        'efficientnet_cifar10': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar10', 'efficientnet_cifar10.pth'),
        }

cifar100_pretrained_paths = {
        'vgg11_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'vgg11_cifar100.pth'),
        'vgg13_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'vgg13_cifar100.pth'),
        'vgg16_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'vgg16_cifar100.pth'),
        'vgg19_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'vgg19_cifar100.pth'),
        'resnet18_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'resnet18_cifar100.pth'),
        'resnet34_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'resnet34_cifar100.pth'),
        'resnet50_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'resnet50_cifar100.pth'),
        'mobilenet_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'mobilenet_cifar100.pth'),
        'mobilenetv2_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'mobilenetv2_cifar100.pth'),
        'efficientnet_cifar100': os.path.join(os.path.dirname(project_dir), 'chkpts', 'cifar100', 'efficientnet_cifar100.pth'),
        }

vocseg_pretrained_paths = {
        'fcn_resnet50_voc_seg': os.path.join(os.path.dirname(project_dir), 'chkpts', 'pascalvoc', 'FCN_ResNet50_Weights.pth'),
        'fcn_resnet101_voc_seg': os.path.join(os.path.dirname(project_dir), 'chkpts', 'pascalvoc', 'FCN_ResNet101_Weights.pth'),
        'deeplabv3_voc_seg': os.path.join(os.path.dirname(project_dir), 'chkpts', 'pascalvoc', 'DeepLabV3_MobileNet_V3_Large_Weights.pth'),
        }

mscoco_pretrained_paths = {
        'ssd300_vgg16_coco': os.path.join(os.path.dirname(project_dir), 'chkpts', 'mscoco', 'SSD300_VGG16_Weights_COCO.pth'),
        'retinanet_resnet50_coco': os.path.join(os.path.dirname(project_dir), 'chkpts', 'mscoco', 'retinanet_resnet50_fpn_COCO.pth'),
        'fasterrcnn_resnet50_coco': os.path.join(os.path.dirname(project_dir), 'chkpts', 'mscoco', 'FasterRCNN_ResNet50_FPN_Weights_COCO.pth'),
        }

pretrained_checkpoint_paths = {
        **cifar10_pretrained_paths,
        **cifar100_pretrained_paths,
        **vocseg_pretrained_paths,
        **mscoco_pretrained_paths,
        }

