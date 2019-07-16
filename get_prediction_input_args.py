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

# Define get_prediction_input_args function below please be certain to replace None
# in the return statement with parser.parse_args() parsed argument 
# collection that you created with this function
# 

def get_prediction_input_args():
    """
    Similiar to get_training_input_args, but for prediction input args
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_image', type=str, default='flowers/test/28/image_05270.jpg', 
                        help='The path to image to predict the flower name and probability')
    parser.add_argument('--checkpoint', type=str, default='classifier.pth', 
                        help='The path to trained model for flower name prediction')
    parser.add_argument('--top_k', type=int, default=3, choices=range(1,103),
                        help='The top (k) matching classes, choose from 1-102')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', 
                        help='The mapping of categories to real names')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='Please select either GPU enabled or disabled by passing --gpu or --no-gpu')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    return parser.parse_args();