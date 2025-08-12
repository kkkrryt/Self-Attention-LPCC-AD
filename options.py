import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='no model',
                        help='model path for pretrain or test')
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    parser.add_argument('--val_list', type=str, default='/data/dataset1/list/base/val.txt',
                        help='val list in train, test list path in test')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--model_type', type=str, default='transformer', help='used in model_entry.py')
    # parser.add_argument('--data_type', type=str, default='scannet', help='used in data_entry.py')
    parser.add_argument('--seed', type=int, default=777, help='set the seed of random functions')
    parser.add_argument('--print_freq', type=int, default=100, help='set the frequency of print information')
    parser.add_argument('--dataset', type=str, default='kitti', help='dataset')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='/data/dataset1/list/base/train.txt')
    parser.add_argument('--batch_size', type=int, default=5120)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0) 
    parser.add_argument('--alpha', type=float, default=0.2)
    # parser.add_argument('--outFeatureDimension', type=int, default=128)
    parser.add_argument('--loss', type=str, default="ce")
    parser.add_argument('--sequence_size', type=int, default=16)
    parser.add_argument('--tree_depth', type=int, default=9)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--lr_schedule', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='', help='dir to store the training result')

    parser.add_argument('--hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--nhead', type=int, default=4, help='number of heads')
    parser.add_argument('--num_layer', type=int, default=6, help='number of encoder layers')

    parser.add_argument('--use_absolute_pos', type=str, default="False")
    parser.add_argument('--use_OctLeFF', type=str, default="False")
    parser.add_argument('--use_OctPEG', type=str, default="False")
    
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


# def get_train_model_dir(args):
#     model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
#     if not os.path.exists(model_dir):
#         os.system('mkdir -p ' + model_dir)
#     args.model_dir = model_dir


# def get_test_result_dir(args):
#     ext = os.path.basename(args.load_model_path).split('.')[0]
#     model_dir = args.load_model_path.replace(ext, '')
#     val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
#     result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
#     if not os.path.exists(result_dir):
#         os.system('mkdir -p ' + result_dir)
#     args.result_dir = result_dir


def save_args(args, save_dir):
    if(os.path.exists(save_dir) == False):
            os.mkdir(save_dir)
    args_path =  os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    # get_train_model_dir(args)
    save_args(args, args.save_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    # get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()