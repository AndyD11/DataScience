#imports
import random
import time
import numpy as np
import pandas as pd
import torchvision
from torchvision import models
import torch
import torch.nn as nn
import cv2
import math
from sklearn.svm import LinearSVC

def img_norm(img):
    return 2 * (np.float32(img) / 255 - 0.5) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(pretrained=True)

class AlexNet_fe(nn.Module):
            def __init__(self):
                super(AlexNet_fe, self).__init__()
                self.features = model.features
                
            def forward(self, x):
                x = self.features(x)
                return x

model = AlexNet_fe()
for param in model.parameters():
    param.requires_grad = False
    
model = model.to(device)

#Load files
#Read in csv with business and corresponding labels
restaurant_to_labels = pd.read_csv("train.csv")
restaurant_to_labels["labels"] = restaurant_to_labels["labels"].fillna("") #Replaces np.nan with empty string
#Read in csv with image and corresponding business
img_to_restaurant = pd.read_csv("train_photo_to_biz_ids.csv")

#Convert businesse_to_labels to a dictionary
restaurants = {}
for index, row in restaurant_to_labels.iterrows():
    restaurants[row["business_id"]] = row["labels"]

restaurant_images = {}
for restaurant in restaurants:
    restaurant_images[restaurant] = []

for index,row in img_to_restaurant.iterrows():
    business_id = row["business_id"]
    photo_id = row["photo_id"]
    filename = "train_photos/" + str(photo_id) + ".jpg"
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64), cv2.INTER_LINEAR)
    img = np.transpose(img, [2, 0, 1])
    img = img_norm(img)
    img = torch.FloatTensor(img).unsqueeze(0)
    img = img.to(device)
    img = torch.Tensor.cpu(model(img)).numpy().flatten()
    restaurant_images.get(business_id).append(img)
    if index%1000 == 0:
        print(str(index) + " images processed!")

def p2(tp,fp,tn,fn):
    return tp/(tp+fp)

def r2(tp,fp,tn,fn):
    return tp/(tp+fn)

def f1_2(p2,r2):
    return ((2*p2*r2)/(p2+r2))

temp = restaurant_to_labels["business_id"].tolist()
random.shuffle(temp)
for i in range(5):
    start_index = i*400
    end_index = (i+1)*400
    training_restaurants = temp[:start_index] + temp[end_index:]
    testing_restaurants = temp[start_index:end_index]
    training_features = []
    training_labels = [[],[],[],[],[],[],[],[],[]]
    for restaurant in training_restaurants:
        for image in restaurant_images.get(restaurant):
            training_features.append(image)
            restaurant_label = restaurants.get(restaurant)
            for j in range(9):
                if str(j) in restaurant_label:
                    training_labels[j].append(1)
                else:
                    training_labels[j].append(0)
    
    svc_list = [LinearSVC().fit(training_features, training_labels[0]), 
            LinearSVC().fit(training_features, training_labels[1]),
            LinearSVC().fit(training_features, training_labels[2]),
            LinearSVC().fit(training_features, training_labels[3]),
            LinearSVC().fit(training_features, training_labels[4]),
            LinearSVC().fit(training_features, training_labels[5]),
            LinearSVC().fit(training_features, training_labels[6]),
            LinearSVC().fit(training_features, training_labels[7]),
            LinearSVC().fit(training_features, training_labels[8])]
    
    p_labels = {}
    for restaurant in testing_restaurants:
        label = [0,0,0,0,0,0,0,0,0]
        for image in restaurant_images.get(restaurant):
            predicted_labels = [svc.predict([image]) for svc in svc_list]
            for k in range(9):
                if predicted_labels[k] == 1:
                    label[k] += 1
        label = [nl/len(restaurant_images.get(restaurant)) for nl in label]
        for nl in range(9):
            if label[nl] < 0.35:
                label[nl] = 0
            else:
                label[nl] = 1
        p_labels[restaurant] = label
    
    ptp = [0,0,0,0,0,0,0,0,0]
    pfp = [0,0,0,0,0,0,0,0,0]
    ptn = [0,0,0,0,0,0,0,0,0]
    pfn = [0,0,0,0,0,0,0,0,0]
    for restaurant in p_labels:
        tl = [0,0,0,0,0,0,0,0,0]
        l = restaurants.get(restaurant)
        for n in range(9):
            if str(n) in l:
                tl[n] = 1
        pl = p_labels.get(restaurant)
        for m in range(9):
            if pl[m] == 1:
                if tl[m] == 1:
                    ptp[m] += 1
                else:
                    pfp[m] +=1
            else:
                if tl[m] == 0:
                    ptn[m] += 1
                else:
                    pfn[m] += 1
    print(ptp)
    print(pfp)
    print(ptn)
    print(pfn)
    ptp = sum(ptp)
    pfp = sum(pfp)
    ptn = sum(ptn)
    pfn = sum(pfn)
    meanf1 = f1_2(p2(ptp,pfp,ptn,pfn), r2(ptp,pfp,ptn,pfn))
    print(meanf1)

    
