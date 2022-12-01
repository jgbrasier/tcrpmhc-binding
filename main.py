import sys
import os
import pandas as pd
import numpy as np

from src.utils import enc_list_bl_max_len, blosum50_20aa

train_file = "tcrpmhc-binding/data_test/sample_train.csv"
test_file = "tcrpmhc-binding/data_test/sample_test.csv"

print('Loading and encoding the data..')
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

encoding = blosum50_20aa

pep_train = enc_list_bl_max_len(train_data.peptide, encoding, 9)
tcra_train = enc_list_bl_max_len(train_data.CDR3a, encoding, 30)
tcrb_train = enc_list_bl_max_len(train_data.CDR3b, encoding, 30)
y_train = np.array(train_data.binder)

pep_test = enc_list_bl_max_len(test_data.peptide, encoding, 9)
tcra_test = enc_list_bl_max_len(test_data.CDR3a, encoding, 30)
tcrb_test = enc_list_bl_max_len(test_data.CDR3b, encoding, 30)

train_inputs = [tcra_train, tcrb_train, pep_train]
test_inputs = [tcra_test, tcrb_test, pep_test]