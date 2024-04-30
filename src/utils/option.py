import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # Model and dataset. 
        parser.add_argument('--n_class', type=int, default=4, help='classification classes')
        parser.add_argument('--train_images', type=str, help='path to train image dataset')
        parser.add_argument('--val_images', type=str, help='path to val image dataset')
        parser.add_argument('--train_csv', type=str, help='path to train csv file')
        parser.add_argument('--val_csv', type=str, help='path to val csv file')
        parser.add_argument('--train_graphs', type=str, help='path to train graph dataset')
        parser.add_argument('--val_graphs', type=str, help='path to val graph dataset')
        parser.add_argument('--model_path', type=str, help='path to where model is saved')
        parser.add_argument('--log_path', type=str, help='path to log files')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--n_epoch', type=int, help='number of epochs')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Adam optiizer hyparameter') # weight_decay=5e-4,4e-3
        parser.add_argument('--gamma', type=float, default=0.1, help='lr scheduler hyperparameter') # gamma=0.1,0.3
        parser.add_argument('--log_interval_local', type=int, default=10, help='classification classes')
        parser.add_argument('--resume', type=str, default="", help='path to saved model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')

        # Parser.
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
