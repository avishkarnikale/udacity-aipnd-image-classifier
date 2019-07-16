#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Avishkar Nikale
# DATE CREATED: 14th July 2019                                  
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#    
#
##
# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_training_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window for training a model.
    This function uses Python's argparse module to create and defined all command line arguments. 
    If the user fails to provide some or all of the arguments, 
    then the default values are used for the missing arguments. 
    
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type=str, default='flowers', 
                        help='The path to data directory')
    parser.add_argument('--save_dir', type=str, default='assets/', 
                        help='The path to save the pytorch checkpoint')
    parser.add_argument('--arch', type=str, default = 'vgg19',
                       help='Please select one of the 2 architectures alexnet or vgg19')
    parser.add_argument('--learning_rate', type=str, default = '0.001',
                       help='Please provide the learning rate')
    parser.add_argument('--hidden_units', type=str, default = '4096',
                       help='Please provide the no of units in the hidden layer (commma seperated)')
    parser.add_argument('--epochs', type=int, default = 5,
                       help='Please provide the number of epochs')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='Please select either GPU enabled or disabled by passing --gpu or --no-gpu')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
                       
    
    return parser.parse_args()