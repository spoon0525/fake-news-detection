from load_dataset import load_data
from train import train, evaluate
from judge import judge
import argparse

parser = argparse.ArgumentParser(description='training for judge real news or fake news')
parser.add_argument('--out_dir', type=str, default="./results", help='path to output folder')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate for train')
parser.add_argument('--train_batch', type=int, default=16, help='batch size for training')
parser.add_argument('--test_batch', type=int, default=16, help='batch size for testing')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train')
parser.add_argument('--weight_decay', type=float, default=0.01, help='value of weight decay')
parser.add_argument('--train_path', type=str, default="GonzaloA/fake_news", help='path to train dataset')
parser.add_argument('--train', type=bool, default=False, help='training the model or not')
parser.add_argument('--eval', type=bool, default=False, help='evaluate this model or not')
parser.add_argument('--pt_path', type=str, default="results\\checkpoint-4500", help='path to pre-train model')
parser.add_argument('--url', type=str, help='the ulr of news to be tested')
args = parser.parse_args()

if args.train:
    train_data, eval_data = load_data(args.train_path)
    model = train(train_data, eval_data, args.lr, args.train_batch, args.test_batch, args.epochs, args.weight_decay, args.out_dir)
    if args.eval:
        evaluate(model)

if args.url != None:
    judge(args.pt_path, args.url)