
import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt


def clean() -> None:
    '''
    Clears the Graph
    '''
    plt.clf()
    plt.cla()
    plt.axis("off")


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, type=str,
                help="name for the saved model")
ap.add_argument("-lr", "--learningrate", required=False,
                default=0.001, type=float, help="learningrate")
ap.add_argument("-b", "--batch", required=False,
                default=4, type=int, help="Batch number")
ap.add_argument("-e", "--epochs", required=False,
                default=20, type=int, help="Epoch cycles")
ap.add_argument("-s","--size",required=False,default=420,type=int,help="Image size")
args = vars(ap.parse_args())

print(args)

# config

FMT = '%H:%M:%S'

vals = args.values()
datasetdir = 'src/dataset/train'
model_name = args['name']
learning_rate = args['learningrate']
runepochs = args['epochs']
btsize = args['batch']

img_height = args['size']
img_width = args['size']
savepath = 'src/EfficientNet/savedmodels'

# model save paths
model_save_path = f'{savepath}/{model_name} model'
weight_save_path = f'{savepath}/configs/{model_name} weight.h5'
json_path = f'{savepath}/configs/{model_name}.json'
plot_path = f'{savepath}/Results/{model_name}.png'
graph_path = f'{savepath}/Results/{model_name} Graph.png'
historypath = f'{savepath}/Results/{model_name} history.json'



if not os.path.isdir(datasetdir):
    print("[WARNING] No Such directory")
    exit()
if not os.path.exists(f'{savepath}'):
    os.makedirs(f'{savepath}')
    os.makedirs(f'{savepath}/configs')
    os.makedirs(f'{savepath}/Results')

console_op_path = f'{savepath}/console op.txt'
console_err = f'{savepath}/console error.txt'

sys.stdout = open(console_op_path, 'w')
sys.stderr = open(console_err, 'w')
START_TIME = datetime.now().strftime(FMT)


# Model
END_TIME = datetime.now().strftime(FMT)
