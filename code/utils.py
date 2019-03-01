import pandas
import csv

LABEL_FILE = '../dataset/daftar-senyawa-beserta-binding-energy.csv'
DATA_FILE = '../dataset/HerbalDB.csv'
OUT_FILE = '../dataset/HerbalDB_labeled.csv'


def label_herbaldb_dataset(label_file=LABEL_FILE, data_file=DATA_FILE, out_file=OUT_FILE):
    '''
    Use label_file (consists of positive data) to label whole data in data_file
    and save the output (labeled data_file) in out_file
    '''
    label_df = pandas.read_csv(label_file, dtype={'Name': str})
    data_df = pandas.read_csv(data_file, dtype={'Name': str})

    out_df = pandas.DataFrame(data=None, columns=data_df.columns)
    out_df['Class'] = 0

    missing = []
    for _, row in label_df.iterrows():
        name = row['Name'] + '.mol'
        energy = row['Binding Energy']
        res = data_df[data_df['Name'] == name]        
        if res.empty:
            # label does not exists in data
            missing.append(row['Name'])
        else:
            out_df.loc[res.index[0]] = res.iloc[0]
            label = 1 if pandas.notna(energy) else 0
            out_df['Class'][res.index[0]] = label

    # save output
    out_df.to_csv(out_file, index=False)
    print(out_df)
    print('\nMissing mol: {}'.format(missing))


# if __name__ == '__main__':
#     label_herbaldb_dataset()