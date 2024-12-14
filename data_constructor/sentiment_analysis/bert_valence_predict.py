import os
import argparse
import logging
import json
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from helpers.TextDataset import TextDataset
from helpers import utils
from helpers.MultiLoss import MultiLoss
from models.MultiBERT import MultiBERT
from transformers import BertTokenizer
import torch.nn
import ast

#print(ast.literal_eval(stt))
class TextDataset(Dataset):
    def __init__(self, data_path, file_list, ):
        # load data
        self.data = pd.DataFrame()
        file_list = file_list.split(",")
        for file in file_list:
            data_df = pd.read_csv(os.path.join(data_path, file + ".csv"))
            data_df.fillna(-1000, inplace=True)
            # data_df = data_df.loc[~data_df.isna().any(1)]
            self.data = self.data.append(data_df, ignore_index=True)
        self.data_dict = self.data.to_dict('index')

    # self.tokenizer = tokenizer
    # self._convert_features(max_seq_length)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        sample = self.data_dict[index]
        return [sample['_id'], 0, 0, sample['content']]

file_appendix = sys.argv[1]
model_dir = sys.argv[2]
model_name = sys.argv[3]
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset_ = TextDataset("../Dict_construct/Final_dict/0402", "Fast_final,Slow_final,Both_final")
my_collator = utils.TextCollator(BertTokenizer.from_pretrained("./pretrain_models/chinese_wwm_ext_pytorch"), 256)
Loader = DataLoader(dataset=dataset_, batch_size=64, shuffle=False, collate_fn=my_collator)

model=torch.load("%s/%s.pkl"%(model_dir,model_name)).to(device)
model.eval()

# writer = csv.writer(open("./predict_result_valence_0911.csv", 'a'))
writer = csv.writer(open("./predict_result_valence_%s.csv"%(file_appendix), 'w'))
writer.writerow(["id", "predict_six", "predict_valence"])
for it in tqdm(Loader):
    pred_six, pred_v = model(it[3].to(device))
    pred_six = torch.nn.Softmax(dim=1)(pred_six).detach().cpu().numpy().round(4)
    pred_v = torch.nn.Softmax(dim=1)(pred_v).detach().cpu().numpy().round(4)
    # pred_six = [round(x, 4) for x in pred_six]
    # pred_v = [round(x, 4) for x in pred_v]
    for iid, six, v in zip(it[0],pred_six,pred_v):
        writer.writerow([iid,str([round(x,4) for x in six]),str([round(x,4) for x in v])])