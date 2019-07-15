# udacity-aipnd-image-classifier

# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program.
In this project, students first develop code for an image classifier built with PyTorch, 
then convert it into a command line application.

## The project has two main files `train.py` and `predict.py` and other supporting files

### Running the train.py 

To run the training first go through the help as below
```
$ python train.py --help
```
This will help you in identifying the arguments needed to train
The program has default set so that it trains even when no arguments are passed
- data_dir defaults to 'flowers' directory
- save_dir defaults to 'assets' directory
- arch defaults to 'vgg19' and the other architecture supported is 'alexnet'
- learning_rate defaults to '0.001' , this might not yeild good results for alexnet architecture
- hidden_units defaults to '4096' , but this can be given multiple values by comma seperation i.e. '4096,512'
- epochs defaults to '5' , 7 will yeild a better result for alexnet
- gpu enabling as given as a --gpu and gpu disabling is --no-gpu , default is gpu enabled

The console will produce outputs for various stages as expected in the rubric
It will print training & validation loss 
It will print the performance on test dataset
The last output on console will produce the path for saved checkpoint of your model
Care has been taken to randomize the name so that you can save multiple results but should not be relied upon
The last output also gives you sample command to run the `predict.py` for the saved checkpoint

### Running the predict.py 

To run the prediction first go through the help as below
```
$ python predict.py --help
```
This will help you in identifying the arguments needed to predict a particular image
The program has default set so that it predicts a default flower from the test set even when no arguments are passed
given that a checkpoint named 'classifier.pth' is present in the present working directory(ImageClassifier) 
which has been created using the `train.py` 
- path_to_image defaults to 'flowers/test/28/image_05270.jpg'
- checkpoint defaults to 'classifier.pth', note that this will fail if you haven't uploaded the checkpoint file
- top_k defaults to 3 for the number of most probable classes of flowers to be shown, choose from 1-102
- category_names defaults to 'cat_to_name.json' in the present working directory(ImageClassifier) 
- gpu enabling as given as --gpu and gpu disabling is --no-gpu , default is gpu enabled

### Supporting files 
- get_input_args - Inspired from the first project in the nanodegree, has 2 methods , 1 for training args and 1 for prediction args
- img_classifier_utils - This helper utils has methods for training, testing, loading checkpoint, processing and predicting any image input
- model_wrapper - This is a wrapper class for pytorch models and helps in decoupling the creation and lifecycle 
- time_elaped - For printing out execution time 

