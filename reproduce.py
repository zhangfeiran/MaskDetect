import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
import utils
import dataset
import metric
import engine
import models.faster_rcnn

try:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.get_device_properties(device)
except:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
model = models.faster_rcnn.get_model(pretrained="coco")
model.to(device)



# if no model, then train one
checkpoint_path = "./data/checkpoint_zfr.pt"
if not os.path.exists(checkpoint_path):
    print('no checpoint_zfr.pt, start training...')
    if not os.path.exists('./data/AIZOO.pkl'):
        dataset.preprocess()
    
    dataset_train = dataset.AIZOODataset("train", dataset.transform_rcnn)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=utils.collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1 # change to 7 or more to fully reproduce, but 1 is enough for the detection of 10 pictures, which can save you some time :)
    for epoch in range(num_epochs):
        engine.train_one_epoch(model, optimizer, None, data_loader_train, device, epoch, print_freq=100)
        lr_scheduler.step()

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
else:
    print('start loading checkpoint_zfr.pt')
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])





# 1 detect
print('start detecting...')

# unmask_idxs = [7844, 7783, 7769, 2029, 6333]
# mask_idxs = [7594, 7722, 7797, 7803, 7842]

for i in range(5):
    img = Image.open(f'./test-images/unmask{i+1}.jpg').convert("RGB")
    img_annoted = engine.detect(model, img, device)
    img_annoted.save(f'./test-images/annoted_unmask{i+1}.jpg')

    img = Image.open(f'./test-images/mask{i+1}.jpg').convert("RGB")
    img_annoted = engine.detect(model, img, device)
    img_annoted.save(f'./test-images/annoted_mask{i+1}.jpg')



