import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import Dataset
from transformers import BertTokenizer


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
        return [sample['id'], sample['label_six'], sample['label_valence'], sample['content']]
    # return {'id':sample['id'], 'label_six':sample['label_six'], 'label_valence':sample['label_valence'],
    # 'input_ids':self.input_ids[index],'input_masks':self.input_masks[index], 'segment_ids':self.segment_ids[index]}

# def _convert_features(self, max_seq_length):
# 	'''
# 	Loads a data file into lists of input_ids, input_mask, and segment_ids.
# 	'''
# 	self.input_ids = []
# 	self.input_masks = []
# 	self.segment_ids = []
# 	for idx in self.data_dict:
# 		content = self.data_dict[idx]['content']
# 		# 分词
# 		tokens = self.tokenizer.tokenize(content)
# 		# tokens进行编码
# 		encode_dict = self.tokenizer.encode_plus(text=tokens,
# 											max_length=max_seq_length,
# 											truncation=True,
# 											padding=True,
# 											is_pretokenized=True,
# 											return_token_type_ids=True,
# 											return_attention_mask=True)
# 		self.input_ids.append(encode_dict['input_ids'])
# 		self.input_masks.append(encode_dict['attention_mask'])
# 		self.segment_ids.append(encode_dict['token_type_ids'])


