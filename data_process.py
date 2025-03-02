import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch

# class PeptideDataset(Dataset):
#     def __init__(self, df, seq_col='PepName', target_col='Taste', target_sheet='1',max_len=50):
#         self.df = df
#         self.seq_col = seq_col
#         self.target_col = target_col
#         self.max_len = max_len
#         self.amino_acid_dict = {
#             'A': 0, 'R': 1, 'D': 2, 'N': 3, 'C': 4,
#             'Q': 5, 'E': 6, 'H': 7, 'I': 8, 'G': 9,
#             'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
#             'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
#         }
# 
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         try:
#             sequence = self.df.loc[idx, self.seq_col]
#             #target = self.df.loc[idx, self.target_col]
#             taste_mapping = {"Umami": 1, "Bitter": 0, 'bitter': 0, 'unami': 1, 'umami':1}
#             target = taste_mapping.get(self.df.loc[idx, self.target_col], self.df.loc[idx, self.target_col])
#             encoded_sequence = self.encode_and_pad(sequence)
#         except:
#             #return torch.tensor(torch.full((50,),20), dtype=torch.long), torch.tensor(0, dtype=torch.float)
#             return torch.full((50,), 20, dtype=torch.long).clone().detach(), torch.tensor(0, dtype=torch.float)
#         return torch.tensor(encoded_sequence, dtype=torch.long), torch.tensor(target, dtype=torch.float)
#
#     def encode_and_pad(self, sequence):
#         encoded_sequence = [self.amino_acid_dict[aa] for aa in sequence if aa in self.amino_acid_dict]
#         padding_len = self.max_len - len(encoded_sequence)
#         padded_sequence = encoded_sequence + [20] * padding_len  # 用 0 进行填充
#         return padded_sequence
#
#     def pad_sequence(self, sequence):
#         padding_len = self.max_len - len(sequence)
#         padding = [[0] * len(self.amino_acids) for _ in range(padding_len)]
#         return sequence + padding
#
#
# def my_collate_fn(batch):
#     batch = [item for item in batch if item is not None]  # 过滤掉None
#     if len(batch) == 0:  # 如果过滤后没有数据，返回None或抛出异常
#         return None
#     # 使用默认方式来合并batch里的数据
#     return torch.utils.data.dataloader.default_collate(batch)

# def create_data_loader(excel_file, batch_size=32, shuffle=True, **kwargs):
#     dataset = PeptideDataset(excel_file, **kwargs)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 使用示例：

class PeptideDataset(Dataset):
    def __init__(self, df, seq_col='PepName', target_col='Taste', target_sheet='1', max_len=50):
        self.df = df
        self.seq_col = seq_col
        self.target_col = target_col
        self.max_len = max_len
        self.amino_acid_dict = {
            'A': 0, 'R': 1, 'D': 2, 'N': 3, 'C': 4,
            'Q': 5, 'E': 6, 'H': 7, 'I': 8, 'G': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        self.feature_cols = [
            "MinEStateIndex", "SMR_VSA1",
            "BCUT2D_MWLOW", "VSA_EState5", "VSA_EState6", "VSA_EState7","PEOE_VSA14","MolLogP"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            sequence = self.df.loc[idx, self.seq_col]
            taste_mapping = {"Umami": 1, "Bitter": 0, 'bitter': 0, 'unami': 1, 'umami': 1}
            target = taste_mapping.get(self.df.loc[idx, self.target_col], self.df.loc[idx, self.target_col])
            encoded_sequence = self.encode_and_pad(sequence)
            features = self.df.loc[idx, self.feature_cols].values.astype(float)
        except:
            return ((torch.full((50,), 20, dtype=torch.long).clone().detach(),
                    torch.zeros(len(self.feature_cols), dtype=torch.float).clone().detach()),
                    torch.tensor(0, dtype=torch.float))

        return ((torch.tensor(encoded_sequence, dtype=torch.long),
                torch.tensor(features, dtype=torch.float)),
                torch.tensor(target, dtype=torch.float))

    def encode_and_pad(self, sequence):
        encoded_sequence = [self.amino_acid_dict[aa] for aa in sequence if aa in self.amino_acid_dict]
        padding_len = self.max_len - len(encoded_sequence)
        padded_sequence = encoded_sequence + [20] * padding_len  # Use 20 to pad
        return padded_sequence

class PeptideDataset1(Dataset):
    def __init__(self, df, seq_col='PepName', target_col='Taste', target_sheet='1', max_len=50):
        self.df = df
        self.seq_col = seq_col
        self.target_col = target_col
        self.max_len = max_len
        self.amino_acid_dict = {
            'A': 0, 'R': 1, 'D': 2, 'N': 3, 'C': 4,
            'Q': 5, 'E': 6, 'H': 7, 'I': 8, 'G': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }

        self.feature_cols = [
            "MinEStateIndex", "SMR_VSA1",
            "BCUT2D_MWLOW", "VSA_EState5", "VSA_EState6", "VSA_EState7","PEOE_VSA14","MolLogP"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            sequence = self.df.loc[idx, self.seq_col]
            taste_mapping = {"Umami": 1, "Bitter": 0, 'bitter': 0, 'unami': 1, 'umami': 1}
            #target = taste_mapping.get(self.df.loc[idx, self.target_col], self.df.loc[idx, self.target_col])
            encoded_sequence = self.encode_and_pad(sequence)
            features = self.df.loc[idx, self.feature_cols].values.astype(float)
        except:
            return ((torch.full((50,), 20, dtype=torch.long).clone().detach(),
                    torch.zeros(len(self.feature_cols), dtype=torch.float).clone().detach()),
                    torch.tensor(0, dtype=torch.float))

        return ((torch.tensor(encoded_sequence, dtype=torch.long),
                torch.tensor(features, dtype=torch.float)),
                #torch.tensor(target, dtype=torch.float)
                )

    def encode_and_pad(self, sequence):
        encoded_sequence = [self.amino_acid_dict[aa] for aa in sequence if aa in self.amino_acid_dict]
        padding_len = self.max_len - len(encoded_sequence)
        padded_sequence = encoded_sequence + [20] * padding_len  # Use 20 to pad
        return padded_sequence

def create_data_loader(excel_file, batch_size=32, shuffle=True, n_splits=5, **kwargs):
    df = pd.read_excel('./dataset/data_processed.xlsx')  # 读取 CSV 文件
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    for train_index, val_index in kf.split(df):
        df_train = df.iloc[train_index]
        df_val = df.iloc[val_index]

        train_dataset = PeptideDataset(df=df_train, **kwargs)
        val_dataset = PeptideDataset(df=df_val, **kwargs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        yield train_loader, val_loader  # 这个 `yield` 在循环内部，不会多次执行
