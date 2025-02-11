import numpy as np
import pandas as pd


import torch

from models.temp_model import CNNModel
from utils import overlapScore, visualize_batch

testX = pd.read_csv('raw_data/testData.csv', sep=',', header=None)
groundTruth = pd.read_csv('raw_data/ground-truth-test.csv', sep=',', header=None)

testX = np.asanyarray(testX)
groundTruth = np.asarray(groundTruth)

model = CNNModel()
model.eval()
model.load_state_dict(torch.load('models/cnn_model.pth'))

with torch.no_grad():
    output = model(torch.Tensor(np.reshape(testX, (len(testX), 1, 100, 100))))
output = output.detach().numpy()
output = output.astype(int)

visualize_batch(
    images=np.reshape(testX, (len(testX),1,100,100)),
    pred_boxes=output,
    gt_boxes=groundTruth,
    save_dir=''
)

score, _ = overlapScore(output, groundTruth)
score /= len(testX)
print('Test Average overlap score : %f' % score)

np.savetxt('results/test-result.csv', output, delimiter=',')