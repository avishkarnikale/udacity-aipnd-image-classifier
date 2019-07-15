import model_wrapper as mw
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from torch.optim import lr_scheduler 
from PIL import Image
import json
import pprint

def train(mw,epochs,gpu,learning_rate):
    print('Training the model with '+str(epochs)+' epochs and with GPU = '+str(gpu))

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    learn_rate = float(learning_rate)
    optimizer = optim.Adam(mw.model.classifier.parameters(), lr=learn_rate)
    mw.model.optimizer = optimizer
    device = "cuda"
    if gpu == False:
        device = "cpu"
        
    mw.model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in mw.dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = mw.model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                mw.model.eval()
                with torch.no_grad():
                    for inputs, labels in mw.dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = mw.model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        # Rubric Section - Validation Loss and Accuracy 
                        # During training, the validation loss and accuracy are displayed                    


                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {test_loss/len(mw.dataloaders['valid']):.3f}.. "
                      f"Valid accuracy: {accuracy/len(mw.dataloaders['valid']):.3f}")
                running_loss = 0
                mw.model.train()
    return
            
def test(mw,gpu):
    print('Test run with GPU = '+str(gpu))
    accuracy = 0
    test_loss = 0
    criterion = nn.NLLLoss()
    device = "cuda"
    if gpu == False:
        device = "cpu"
    mw.model.eval()
    with torch.no_grad():
        for inputs, labels in mw.dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = mw.model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Test accuracy: {accuracy/len(mw.dataloaders['test']):.3f}"
                 )
    mw.model.train()
    return

def loadModel(checkpoint,category_names):
        print('Loading checkpoint from '+checkpoint)
        model_2_return = None
        chpt = torch.load(checkpoint)
        
        model_2_return  =  mw.ModelWrapper(chpt['arch'])
        model_2_return.freeze_params()
        model_2_return.hidden_layers = chpt['classifier_layers'][1:len(chpt['classifier_layers'])-1]
        model_2_return.create_classifier(get_layers_as_comma_sep_string(model_2_return.hidden_layers))
        model_2_return.class_to_idx = chpt['class_to_idx']
        model_2_return.imagenet_means = chpt['imagenet_means']
        model_2_return.imagenet_stdevs = chpt['imagenet_stdevs']
        model_2_return.model.load_state_dict(chpt['state_dict'])
        model_2_return.model.optimizer = optim.Adam(model_2_return.model.classifier.parameters(), lr=float(chpt['lr']))
        model_2_return.model.optimizer.load_state_dict(chpt['optimizer_state_dict'])
        with open(category_names, 'r') as f:
            model_2_return.cat_to_name = json.load(f)
        return model_2_return
 
def get_layers_as_comma_sep_string(hidden_layers):
    retVal = ""
    count = 0 
    for i in hidden_layers:
        if len(hidden_layers) == 1:
            return str(hidden_layers[0])
        if count < len(hidden_layers):
            retVal = retVal + str(i) + ','
        else:
            retVal = retVal + str(i) 
        count = count + 1
    print(retVal)
    return retVal


def process_image(image,imagenet_means,imagenet_stdevs):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Tensor
    '''
    
    
    # Use transform for coverting to image tensor :-) , 
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_means, imagenet_stdevs)
    ])
    img_tensor = transform_image(image)

    img_tensor.unsqueeze_(0) 
    return img_tensor

def predict(image_path, mw, top_k,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('Prediction run with GPU = '+str(gpu))
    img = Image.open(image_path)

    # Convert image to a tensor 

    img = process_image(img,mw.imagenet_means,mw.imagenet_stdevs)
   
    device = "cuda"
    if gpu == False:
        device = "cpu"
        img = img.cpu()
    else:
        img = img.cuda()
    mw.model.to(device)
    mw.model.eval()
    # Send the image tensor in a forward pass and get the prob dist
    probs = torch.exp(mw.model.forward(img))
    
    # Using the top_k arg get the top probabilities and labels
    top_probs, top_labs = probs.topk(top_k)
    #Bring to CPU if GPU is enabled
    if gpu == True:
        top_probs = top_probs.cpu()
        top_labs = top_labs.cpu()
    # Do a numpy conversion 
        
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes reversing the json read 
    idx_to_class = {val: key for key, val in    
                                      mw.class_to_idx.items()}
    top_label_names = [idx_to_class[lab] for lab in top_labs]
    top_flower_names = [mw.cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_label_names, top_flower_names
