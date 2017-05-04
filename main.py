import argparse
import os
from util import *
import tensorflow as tf
from DiscoGAN import DiscoGAN
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=5, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for DiscoGAN')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term (beta 1) of adam')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='momentum term (beta 2) of adam')
parser.add_argument('--save_iteration_freq', dest='save_iteration_freq', type=int, default=500, help='save a model every save_epoch_iteration')
parser.add_argument('--check_results_freq', dest='check_results_freq', type=int, default=500, help='check a intermediate result per check_results_freq')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='plot represented intermediate result is saved')
parser.add_argument('--is_train', dest='is_train', default=False, help='train the model for train set')
parser.add_argument('--is_test', dest='is_test', default=True, help='Import model from saved model, and test the model for test set')
parser.add_argument('--is_load', dest='is_load', default=False, help='')
parser.add_argument('--L1_lambda', dest='L1_lambda', default=100, help='Lambda')
args = parser.parse_args()

def main(_):
    if(os.path.isdir(args.checkpoint_dir) == False) :
        os.mkdir(args.checkpoint_dir)
    if (os.path.isdir(args.result_dir) == False):
        os.mkdir(args.result_dir)
    else :
        file_clear(args.result_dir)

    A_data_with_class, A_data = generate_data(13, 3, radius=2, center=(-2, -2))
    B_data_with_class, B_data = generate_data(13, 3, radius=2, center=(2, 2))

    train_data = (A_data, B_data)
    test_data = (A_data_with_class, B_data_with_class)

    with tf.Session() as sess:
        model = DiscoGAN(sess,args)
        if(args.is_train) :
            model.train(args,train_data)
        if(args.is_test):
            if(model.load(args.checkpoint_dir)) :
                model.test(args,test_data)

if __name__ == '__main__':
    tf.app.run()
