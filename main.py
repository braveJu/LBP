import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from utils import HP, LABEL_DICT, get_KTH, preprocess
from models import DongjuModel
from dataset import KTHDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline




# def train_loop(dataloader, model, loss_fn, optimizer, device):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         loss, current = loss.item(), (batch+1) * len(X)
#         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test_loop(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# def inference(dataloader):
#     model = DongjuModel()
#     model.load_state_dict(torch.load("dongju_model_weights.pth"))
#     model.eval()
#     cnt = 0
#     for x in dataloader:
#         if cnt == 10:
#             break
#         cnt += 1
#         res = LABEL_DICT[model(x).argmax(1)]
#     print(res)
    
    
    
# if __name__ == '__main__':
#     df = pd.read_csv('valid.txt')
#     cuda_available = torch.cuda.is_available()
#     device = torch.device("cuda" if cuda_available else "cpu")
    
#     dataset = KTHDataset(df['file_name'], df['label'])
#     train_dataloader = DataLoader(dataset, batch_size=HP['batch_size'], shuffle=True)
#     valid_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
#     # print(inference(valid_dataloader))
#     model = DongjuModel(output_size=HP['output_size']).to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=HP['learning_rate'])
    
#     for t in range(HP['epochs']):
#         print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(train_dataloader, model, loss_fn, optimizer, device)
#         test_loop(valid_dataloader, model, loss_fn)
#     torch.save(model.state_dict(), "dongju_model_weights1.pth")
#     print("Done!")




if __name__ =='__main__':

    X_train, X_test, y_train, y_test = get_KTH()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)
    print(f"정확도 : {sum(pred_y == y_test)/len(pred_y)}")

