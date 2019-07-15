import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import pprint
import random

supported_models = ['alexnet','vgg19']

class ModelWrapper():
        
    def __init__(self,arch):
        super().__init__()
        if (arch not in supported_models):
            raise Exception('Arch not supported')

        self.model = None
        if(arch == 'vgg19'):
            self.model = models.vgg19(pretrained=True)
            self.in_features = 25088
        elif(arch == 'alexnet'):
            self.model = models.alexnet(pretrained=True)
            self.in_features = 9216
        else:
            print('Arch not supported , please choose either alexnet or vgg19')
            
        self.out_features = 102
        return
    
    def setup_data(self,data_dir):
        print('setup_data - data_dir is ',data_dir)
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        self.imagenet_means = [0.485, 0.456, 0.406]
        self.imagenet_stdevs = [0.229, 0.224, 0.225]
        return
    
    def get_base_transforms(self):
        return [transforms.CenterCrop(224),transforms.ToTensor(),
                transforms.Normalize(self.imagenet_means,self.imagenet_stdevs )]


    def get_training_spec_transforms(self):
        t = [transforms.RandomResizedCrop(244),transforms.RandomHorizontalFlip()]
        t.extend(self.get_base_transforms())
        return t
    
    def augment_normalize_data(self):
        print('augment_training_data')
        train_transform = transforms.Compose(self.get_training_spec_transforms())
        valid_transform = transforms.Compose(self.get_base_transforms())
        test_transform = transforms.Compose(self.get_base_transforms())
        self.data_transforms = {'train': train_transform,'valid':valid_transform,'test':test_transform}
        return
    
    def load_data(self):
        print('load_data')
        train_ds = datasets.ImageFolder(self.train_dir,   transform=self.data_transforms['train']) 
        valid_ds = datasets.ImageFolder(self.valid_dir,   transform=self.data_transforms['valid']) 
        test_ds = datasets.ImageFolder(self.test_dir,   transform=self.data_transforms['test']) 

        self.image_datasets = {'train': train_ds,'valid':valid_ds,'test':test_ds}
        
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=64, shuffle=True)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=True)

        self.dataloaders = {'train': train_dl,'valid':valid_dl,'test':test_dl}
        return
    
    def freeze_params(self):
        print('Freeze model parameters so we don\'t backprop through them')
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False
        return
    
    def create_classifier(self,hidden_layers):
        fcstr = 'fc'
        relustr = 'relu'
        dropoutstr = 'dropOut'
        layers = hidden_layers.split(",")
        list = [self.in_features]
        list.extend(layers)
        list.extend([self.out_features])
        list = [int(i) for i in list] 
        od = OrderedDict() 
        count = 0
        for i in list:
            count = count + 1
            if count == 1:          
                od[fcstr + str(count)] = nn.Linear(list[count-1], list[count])
                od[relustr + str(count)] = nn.ReLU()
                od[dropoutstr+  str(count)] = nn.Dropout(0.35)
            elif count == len(list):
                od['output'] = nn.LogSoftmax(dim=1)
            elif count == (len(list) - 1):
                od[fcstr + str(count)] = nn.Linear(list[count-1], list[count])
            elif count > 1 and count < len(list):
                od[fcstr + str(count)] = nn.Linear(list[count-1], list[count])
                od[relustr + str(count)] = nn.ReLU()
                od[dropoutstr+  str(count)] = nn.Dropout(0.35)
        #print(od)
        self.classifier_layers = list
        self.model.classifier = nn.Sequential(od)
        return
    
    def save(self,arch,epochs,learn_rate,save_dir):
        print(' Saving the trained model')
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        self.model.cpu()
        path_to_save = save_dir+arch+'_'+str(random.randint(1,10001))+'_classifier.pth'
        torch.save({'arch': arch,
                    'epochs': epochs,
                    'lr': learn_rate,
                    'imagenet_means' : self.imagenet_means,
                    'imagenet_stdevs' : self.imagenet_stdevs,
                    'input_size': self.in_features,
                    'output_size':self.out_features,
                    'classifier_layers': self.classifier_layers,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                    'class_to_idx': self.model.class_to_idx}, 
                    path_to_save)
        print('Your trained model is kept in "'+path_to_save+'"')
        print('Please use the above path as ->  python predict.py --checkpoint '+path_to_save)
        return
    
    def load(self):
        print('load')
        return
    
    