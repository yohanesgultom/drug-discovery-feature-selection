'''
Get the output files of PaDEL descriptor software and turn them into dataset:
1. PubChem Active HIV-1 Protease: positive data
2. Decoys: negative data
3. HerbalDB (labeled): manually labelled data (positive + negative)

@author yohanes.gultom@gmail.com
'''

import pandas
import csv

output_file = '../dataset/dataset.csv'
output_test_file = '../dataset/dataset_test.csv'
output_test_expanded_file = '../dataset/dataset_test_expanded.csv'

# File dataset
drug_file = '../dataset/pubchem-compound-active-hiv1-protease.csv'
decoy_file = '../dataset/decoys.csv'
herbal_file = '../dataset/HerbalDB_labeled.csv'
herbal_expanded_file = '../dataset/HerbalDB_labeled_expanded.csv'

# Baca file
drug = pandas.read_csv(drug_file, dtype={'Name': str}, index_col=0)
decoy = pandas.read_csv(decoy_file, dtype={'Name': str}, index_col=0)
herbal = pandas.read_csv(herbal_file, dtype={'Name': str}, index_col=0)
herbal_expanded = pandas.read_csv(herbal_expanded_file, dtype={'Name': str}, index_col=0)

# Menentukan kelas
drug['Class'] = 1
decoy['Class'] = 0

# Mengambil decoy sejumlah drug secara random, sehingga dataset menjadi balance
jml_drug = len(drug)
decoy = decoy.drop_duplicates()
decoy_subset = decoy.sample(n=jml_drug)

# Menggabungkan kedua dataset dan mengisi nilai yg kosong dengan 0
dataset = pandas.concat([drug, decoy_subset]).sample(frac=1)
# dataset = pandas.concat([drug, decoy]).sample(frac=1)
dataset.fillna(value=0, inplace=True)
herbal.fillna(value=0, inplace=True)
herbal_expanded.fillna(value=0, inplace=True)

# Simpan hasilnya untuk digunakan pada langkah selanjutnya
dataset.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
herbal.to_csv(output_test_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
herbal_expanded.to_csv(output_test_expanded_file, index=False, quoting=csv.QUOTE_NONNUMERIC)