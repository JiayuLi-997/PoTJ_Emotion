import os
import argparse
import logging
import json
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from helpers.TextDataset import TextDataset
from helpers import utils
from helpers.MultiLoss import MultiLoss
from models.MultiBERT import MultiBERT
from transformers import BertTokenizer

INF = 100000000
logger = logging.getLogger()


def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

'''
python -u main.py --gpu 0 --log_file ./logs/0811_test.log --train_files usual_all,virus_notest --val_files virus_test --test_files weibo_test --pretrain_path ./pretrain_models/chinese_wwm_ext_pytorch --hidden_layers 32 16 --batch_size 32 --lr 0.001 --alpha 0.5 --test_epoch 10

python -u main.py --gpu 0 --log_file ./logs/0818_test.log --train_files augmented --val_files virus_test --test_files weibo_test --pretrain_path ./pretrain_models/chinese_wwm_ext_pytorch --hidden_layers 200 20 --batch_size 32 --lr 0.0014 --alpha 0.5 --test_epoch 10
'''

#TODO: batch_size, lr, hidden_layers, alpha
def add_argument(parser):
    parser.register('type', 'bool', str2bool)

    # General settings
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--log_file", type=str, default="./logs/test.log", help="path for log file.")

    # input
    parser.add_argument("--data_path", type=str, default="../data/")
    parser.add_argument("--train_files", type=str, default="usual_all,virus_train",
                        help="List of training files, split with comma.")
    parser.add_argument("--val_files", type=str, default="virus_test",
                        help="List of validation files, split with comma.")
    parser.add_argument("--test_files", type=str, default="weibo_test", help="List of test files, split with comma.")

    # training settings
    parser.add_argument("--model", type=str, default='MultiBERT')
    parser.add_argument("--pretrain_path", type=str, default='./pretrain/')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=1, help="how often test the metrcis on test set.")
    parser.add_argument("--train_test_epoch", type=int, default=10, help="how often test the metrics on training set.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_iters", type=int, default=1, help="进行多少random seed的重复训练&测试，多次训练以测试模型的稳定性")
    parser.add_argument("--finetune_bert", type='bool', default='False', help="是否对BERT进行finetune")

    # model settings
    parser.add_argument("--hidden_layers", nargs="+", help="List of hidden layers.")
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--max_seq_length", type=int, default=256)

    # Result saving
    parser.add_argument("--cp_path", type=str, default="./checkpoints/")
    parser.add_argument("--save_path", type=str, default="./predict_results/")
    parser.add_argument("--result_anno", type=str, default="test")

    parser.add_argument("--activate", type=str, default="tanh")
    parser.add_argument("--weight", type=str, default="false")
    return parser


def get_mainargs(args):
    arg_dict = vars(args)
    main_list = []
    for k in arg_dict:
        if k in ["gpu", "data_path", "split_path", "rating_level", "model", "save_path", "log_file", "cp_path",
                 "pretrain_path", 'test_files', 'val_files', 'train_files', 'result_anno', 'activate']:
            continue
        v = arg_dict[k]
        if type(v) == float:
            v = "%.6f" % (v)
        main_list.append(str(k) + "=" + str(v))
    return ",".join(main_list)


def train(net, data_loaders, optimizer, criterion, args, device, prefix):
    os.makedirs(os.path.join(args.cp_path, prefix), exist_ok=True)
    min_val_loss = INF
    earlystop_iters = 0
    stop_epoch = 0
    tol = 1e-4
    # run!
    logging.info("Start training!")
    try:
        for epoch in range(args.max_epoch):
            train_loss, train_num = 0, 0
            net.train()
            for data in tqdm(data_loaders['train']):
                label_six, label_v = [d.to(device) for d in data[1:-1]]
                ids, tokens = data[0], data[-1]
                for key in tokens:
                    tokens[key] = tokens[key].to(device)
                # forward
                pred_six, pred_v = net(tokens)
                # backward
                loss = criterion(label_six, label_v, pred_six, pred_v)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_num += pred_six.shape[0]
            train_loss /= train_num

            # predict
            val_loss, val_metrics = evaluate(net, data_loaders['val'], criterion, device, phase='val', save_result=True,
                                             prefix=prefix)
            # early stop
            if val_loss < min_val_loss - tol:
                logger.info("[epoch %d] Train Loss=%.4f, Min val loss from %.4f to %.4f" % (
                epoch, train_loss, min_val_loss, val_loss))
                min_val_loss = val_loss
                earlystop_iters = 0
                stop_epoch = epoch
                logger.info("Save network to %s" % (os.path.join(args.cp_path, "%s_best.pkl" % (prefix))))
                torch.save(net, os.path.join(args.cp_path, "%s_best.pkl" % (prefix)))
            else:
                logger.info("[epoch %d] Train Loss=%.4f, Val Loss=%.4f (Min loss=%.4f in epoch %d)" % (
                epoch, train_loss, val_loss, min_val_loss, stop_epoch))
                earlystop_iters += 1
            output_val = "Val -- "
            for item in val_metrics:
                output_val += item[0] + ":" + "%.3f" % (item[1]) + ", "
            logger.info(output_val)
            if earlystop_iters >= args.patience:
                break
            # test results
            if epoch % args.test_epoch == 0:
                test_loss, test_metrics = evaluate(net, data_loaders['test'], criterion, device, phase='test',
                                                   save_result=True, prefix=prefix)
                output_test = "Test -- "
                for item in test_metrics:
                    output_test += item[0] + ":" + "%.3f" % (item[1]) + ", "
                logger.info("[epoch %d] Train loss: %.3f, Val loss: %.3f, Test loss: %.3f (earlystop: %d)" % (
                epoch, train_loss, val_loss, test_loss, earlystop_iters))
                logger.info(output_test)
            # inputs = 'pred_six'
            # while inputs != 'continue':
            # 	try:
            # 		print(eval(inputs))
            # 	except Exception as e:
            # 		print(e)
            # 	inputs = input()
            if epoch % args.train_test_epoch == 0:
                train_loss, train_metrics = evaluate(net, data_loaders['train'], criterion, device, phase='train',
                                                     save_result=False, prefix=prefix)
                output_train = "Train -- "
                for item in train_metrics:
                    output_train += item[0] + ":" + "%.3f" % (item[1]) + ", "
                logger.info(output_train)
    except KeyboardInterrupt:
        logging.info("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
            exit(1)

    logger.info("Training done! Best iter=%d" % (stop_epoch))
    net = torch.load(os.path.join(args.cp_path, "%s_best.pkl" % (prefix)))
    return net


def evaluate(net, data_loader, criterion, device, phase='test', save_result=False, prefix=""):
    test_loss, test_num = 0, 0
    test_label_six, test_label_v, test_pred_six, test_pred_v = [], [], [], []
    net.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            label_six, label_v = [d.to(device) for d in data[1:-1]]
            ids, tokens = data[0], data[-1]
            for key in tokens:
                tokens[key] = tokens[key].to(device)
            pred_six, pred_v = net(tokens)
            loss = criterion(label_six, label_v, pred_six, pred_v)
            test_pred_six += pred_six.cpu().detach().tolist()
            # test_pred_v += pred_v.cpu().detach().view(-1).tolist()
            test_pred_v += pred_v.cpu().detach().tolist()
            test_label_six += label_six.cpu().view(-1).tolist()
            test_label_v += label_v.cpu().view(-1).tolist()
            test_loss += loss.item()
            test_num += pred_six.shape[0]
        test_loss /= test_num
    test_metrics = utils.get_metrics(test_label_six, test_label_v, test_pred_six, test_pred_v)  # [(metric, value)]
    if save_result:
        np.save(os.path.join(args.cp_path, prefix, "%s_label_six.npy" % (phase)), np.array(test_label_six), )
        np.save(os.path.join(args.cp_path, prefix, "%s_label_v.npy" % (phase)), np.array(test_label_v), )
        np.save(os.path.join(args.cp_path, prefix, "%s_pred_six.npy" % (phase)), np.array(test_pred_six), )
        np.save(os.path.join(args.cp_path, prefix, "%s_pred_v.npy" % (phase)), np.array(test_pred_v), )
    return test_loss, test_metrics


def train_eval(args, tokenizer, seed, device):
    # fix the random seed
    set_all_seeds(seed)
    prefix = get_mainargs(args) + ",seed=%d" % (seed)
    # loading data
    logger.info("Training with seed=%d" % (seed))
    dataset_dict = {}
    data_loaders = {}
    shuffle = {'train': True}
    logger.info("Loading data...")
    my_collator = utils.TextCollator(tokenizer, args.max_seq_length)
    for phase in ['train', 'val', 'test']:
        dataset_dict[phase] = TextDataset(args.data_path, getattr(args, '%s_files' % (
            phase)))  # ,tokenizer=tokenizer,max_seq_length = args.max_seq_length)
        logging.info("Length of %s set: %d" % (phase, len(dataset_dict[phase].data)))
        data_loaders[phase] = DataLoader(dataset=dataset_dict[phase], batch_size=args.batch_size,
                                         shuffle=shuffle.get(phase, False), collate_fn=my_collator)
    # define the model
    logger.info("Loading model...")
    net = eval(args.model)(args.pretrain_path, args.hidden_layers, args.dropout_prob, args.finetune_bert, args.activate == "relu")
    net = net.to(device)
    logger.info(str(net))

    # define optimizer and criterion
    logger.info("Set MultiLoss with alpha=%.2f" % (args.alpha))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.l2)
    criterion = MultiLoss(args.alpha, weight=(args.weight=="true" or args.weight=="True"))

    # training
    net = train(net, data_loaders, optimizer, criterion, args, device, prefix)

    # evaluate
    logging.info("Final evalation: ")
    for phase in ['train', 'val', 'test']:
        loss, metrics = evaluate(net, data_loaders[phase], criterion, device, phase)
        output = ""
        for item in metrics:
            output += item[0] + ":" + "%.3f" % (item[1]) + ", "
        if phase == 'test':
            with open("result.txt", "a") as file:
                file.write(prefix + " " + phase + "\n")
                file.write(output + "\n")
                file.write(f"Loss={loss}\n\n\n")
        logging.info("%s loss= %.3f" % (phase, loss))
        logging.info("%s -- " % (phase) + output)


def run(args):
    # set device
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info("Training with device %s" % (str(device)))
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    logger.info("Load tokenizer from %s" % (args.pretrain_path))
    for seed in range(args.max_iters):
        train_eval(args, tokenizer, seed, device)
    # if torch.cuda.is_available() and args.gpu>-1:
    # 	torch.cuda.empty_cache()
    # inputs = "cuda"
    # while inputs != 'continue':
    # 	try:
    # 		print(eval(inputs))
    # 	except Exception as e:
    # 		print(e)
    # 	inputs = input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument(parser)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %I:%M:%S %p')
    logfile = logging.FileHandler(args.log_file, 'a')

    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    logger.info("Save log to file: %s" % (args.log_file))
    if not args.model in args.cp_path:
        args.cp_path = os.path.join(args.cp_path, args.model)
    os.makedirs(args.cp_path, exist_ok=True)
    args.hidden_layers = [int(x) for x in args.hidden_layers]

    run(args)