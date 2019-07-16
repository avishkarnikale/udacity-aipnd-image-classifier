import model_wrapper as mw
from get_training_input_args import get_training_input_args
from img_classifier_utils import train,test
from time_elapsed import TimeElapsed

# Start Training 
in_arg = get_training_input_args()
model =  mw.ModelWrapper(in_arg.arch)
model.setup_data(in_arg.data_dir)
model.augment_normalize_data()
model.load_data()
model.freeze_params()
model.create_classifier(in_arg.hidden_units)
te = TimeElapsed('Training')
train(model,in_arg.epochs,in_arg.gpu,in_arg.learning_rate)
te.stop_and_print_time_elapsed()
te = TimeElapsed('Test')
test(model,in_arg.gpu)
te.stop_and_print_time_elapsed()
model.save(in_arg.arch,in_arg.epochs,in_arg.learning_rate,in_arg.save_dir)

