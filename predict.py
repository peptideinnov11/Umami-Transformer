import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_process import PeptideDataset1
from model.model import TransformerModel
import pandas as pd

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
model_dim = 8
model1 = TransformerModel(50, model_dim, 8, 128, 3, 2).to(device)
model2 = TransformerModel(50, model_dim, 8, 128, 3, 2).to(device)
model3 = TransformerModel(50, model_dim, 8, 128, 3, 2).to(device)
model4 = TransformerModel(50, model_dim, 8, 128, 3, 2).to(device)
model5 = TransformerModel(50, model_dim, 8, 128, 3, 2).to(device)

model1.load_state_dict(torch.load('final/best_model_fold_0.pth'))
model2.load_state_dict(torch.load('final/best_model_fold_1.pth'))
model3.load_state_dict(torch.load('final/best_model_fold_2.pth'))
model4.load_state_dict(torch.load('final/best_model_fold_3.pth'))
model5.load_state_dict(torch.load('final/best_model_fold_4.pth'))

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

data = pd.read_csv('./input.csv')  # 替换为您的 CSV 文件名
data1 = {}
data1['PepName'] = data['PepName']
train_set = PeptideDataset1(df=data)
val_loader = DataLoader(train_set, batch_size=1, shuffle=False)
results = []
for i in tqdm(val_loader):
    result1 = model1(i[0])
    result2 = model2(i[0])
    result3 = model3(i[0])
    result4 = model4(i[0])
    result5 = model5(i[0])
    results.append((result1.item()+result2.item()+result3.item()+result4.item()+result5.item())/5)

data1['Inference_Result'] = pd.Series(results)
data1['Inference_Result'] = data1['Inference_Result'].apply(lambda x: 'umami' if x > 0.5 else 'bitter')

pred = pd.DataFrame.from_dict(data1)
pred.to_csv('result/result.csv', index=False)  # 保存结果到 CSV 文件

