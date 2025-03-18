from torch import nn
import os
from transformers import BertModel
import torch.nn.functional as F
import torch


class MultiBERT(nn.Module):
    def __init__(self, model_dir, hidden_layers=[32, 16], dropout_prob=0.5, finetune=False, relu=False):
        super(MultiBERT, self).__init__()

        # load pretrain model
        assert os.path.exists(model_dir), 'Error: pretrained model not exist! '
        self.bert = BertModel.from_pretrained(model_dir)
        self.bert_config = self.bert.config

        if not finetune:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout_prob)

        # define MLP
        bert_outdim = self.bert_config.hidden_size
        self.hidden_layers = [bert_outdim] + hidden_layers
        self.embedding = nn.Sequential()
        for i in range(len(self.hidden_layers) - 1):
            self.embedding.add_module('Linear-%d' % (i), nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            if relu:
                self.embedding.add_module('Activation-%d'%(i),nn.ReLU())
            else:
                self.embedding.add_module('Activation-%d' % (i), nn.Tanh())

        # output layer
        self.predictor_six = nn.Linear(self.hidden_layers[-1], 6)
        self.predictor_v = nn.Linear(self.hidden_layers[-1], 3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self._init_params()

    def _init_params(self):
        std = 1
        mean = 0
        for m in self.embedding:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                nn.init.normal_(m.bias.data, mean=mean, std=std)
        nn.init.normal_(self.predictor_six.weight.data, mean=mean, std=std)
        nn.init.normal_(self.predictor_six.bias.data, mean=mean, std=std)
        nn.init.normal_(self.predictor_v.weight.data, mean=mean, std=std)
        nn.init.normal_(self.predictor_v.bias.data, mean=mean, std=std)

    def forward(self, tokens):
        #print(tokens)
        bert_outputs = self.bert(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            token_type_ids=tokens['token_type_ids'],
        )
        # seq_out, pooled_out = bert_outputs[0], bert_outputs[1]
        ## 对反向传播backward截断
        # bert_output = self.dropout(pooled_out.detach())
        seq_out, bert_output = bert_outputs[0], bert_outputs[1]  # 将grad设置为false后，不会再回传

        # get output
        mlp_output = self.embedding(self.dropout(bert_output))  # batch_size * hidden_layers[-1]
        pred_six = self.predictor_six(mlp_output)
        pred_v = self.predictor_v(mlp_output)

        return pred_six, pred_v




