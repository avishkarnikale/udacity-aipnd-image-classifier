from get_prediction_input_args import get_prediction_input_args
from img_classifier_utils import loadModel,get_layers_as_comma_sep_string,predict

# Start Training 
in_arg = get_prediction_input_args()

model =  loadModel(in_arg.checkpoint,in_arg.category_names)

probs, labs, flowers = predict(in_arg.path_to_image,model,in_arg.top_k,in_arg.gpu)

for i in range(1,in_arg.top_k+1):
    print()
    print('------------------------------------------------------------')
    print('The name at position '+str(i)+'  possible match is -> '+flowers[i-1])
    print('The probability of this being correct is -> '+str(probs[i-1]))
    print('The confidence can be assumed to be -> '+str(round(probs[i-1]*100))+'%')
    print('------------------------------------------------------------')
