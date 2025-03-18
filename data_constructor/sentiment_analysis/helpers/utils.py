import numpy as np
import pandas as pd
import os
import sys
from sklearn import metrics
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import datetime

class TextCollator:
	def __init__(self, tokenizer,max_seq_length):
		self.tokenizer = tokenizer
		self.max_seq_length = max_seq_length

	def __call__(self, batch):
		ids_list, label_six_list, label_valence_list = [],[],[]
		batch_sentences = []
		for id, label_six, label_v, content in batch:
			ids_list.append(id)
			label_six_list.append(label_six)
			# label_valence_list.append(label_v)
			label_valence_list.append(label_v+1)
			batch_sentences.append(content)
		label_six_list = torch.LongTensor(label_six_list).view(-1,1)
		# label_valence_list = torch.FloatTensor(label_valence_list).view(-1,1)
		label_valence_list = torch.LongTensor(label_valence_list).view(-1,1)
		#print(ids_list)
		#print(batch_sentences)
		tokens = self.tokenizer(batch_sentences,padding=True,truncation=True,max_length=self.max_seq_length,return_tensors="pt")
		tokens['label'] = label_valence_list
		return [ids_list,label_six_list,label_valence_list,tokens]

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_metrics(label_six,label_v,pred_six,pred_v,threshold=[0.33,0.67],select_metrics=['six_acc','six_f1','six_each_f1','v_acc','v_f1','v_each_f1']):
	all_results = []
	try:
		pred_six_idx = np.argmax(pred_six,axis=1)
		if 'six_acc' in select_metrics:
			all_results.append(['six_acc',metrics.accuracy_score(label_six,pred_six_idx)])
		if 'six_f1' in select_metrics:
			all_results.append(['six_f1',metrics.f1_score(label_six,pred_six_idx,average='macro')])
		# pred_v_idx = np.digitize(pred_v,threshold) - 1
		pred_v_idx = np.argmax(pred_v,axis=1)
		label_v = np.array(label_v).astype(int)
		pred_v_idx = pred_v_idx[(label_v>-1)&(label_v<3)]
		label_v_idx = label_v[(label_v>-1)&(label_v<3)].astype(int)
		if 'v_acc' in select_metrics:
			all_results.append(['v_acc',metrics.accuracy_score(label_v_idx,pred_v_idx)])
		if 'v_f1' in select_metrics:
			all_results.append(['v_f1',metrics.f1_score(label_v_idx,pred_v_idx,average='macro')])
		if 'six_each_f1' in select_metrics:
			f1_class = metrics.f1_score(label_six,pred_six_idx,average=None)
			for i,f1 in enumerate(f1_class):
				all_results.append(['six_f1_%d'%(i),f1])
		if 'v_each_f1' in select_metrics:
			f1_class = metrics.f1_score(label_v_idx,pred_v_idx,average=None)
			for i,f1 in enumerate(f1_class):
				all_results.append(['v_f1_%d'%(i),f1])
			# inputs = 'f1_class'
			# while inputs != 'continue':
			# 	try:
			# 		print(eval(inputs))
			# 	except Exception as e:
			# 		print(e)
			# 	inputs = input()

	except Exception as e:
		print("Error in metrics: ",e)
		inputs = 'label_v[:10]'
		while inputs != 'continue':
			try:
				print(eval(inputs))
			except Exception as e:
				print(e)
			inputs = input()
	return all_results
