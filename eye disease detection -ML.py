#!/usr/bin/env python
# coding: utf-8

# In[37]:


pip install scikit-learn


# In[2]:


pip install tensorflow


# In[3]:


pip install tensorflow-gpu


# In[4]:


pip install scikit-learn matplotlib


# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import random
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# In[6]:


import os
for dirname, _, filenames in os.walk("C:\\Users\\Downloads\\archive (7)"):
    for filename in filenames:
        os.path.join(dirname, filename)
df = pd.read_csv("C:\\Users\\Downloads\\archive (7)\\full_df.csv")


# In[7]:


df.head()


# In[8]:


#making the target dataframes from the doctor's comments 
def has_armd(text):
    if "age-related macular degeneration" in text:
        return 1
    else:
        return 0
df["left_armd"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_armd(x))
df["right_armd"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_armd(x))

left_armd = df.loc[(df.A ==1) & (df.left_armd == 1)]["Left-Fundus"].values
print(left_armd[:10])
right_armd = df.loc[(df.A ==1) & (df.right_armd == 1)]["Right-Fundus"].values
print(right_armd[:10])


# In[9]:


print("Number of images in left cataract: {}".format(len(left_armd)))
print("Number of images in right cataract: {}".format(len(right_armd)))


# In[10]:


#now taking same amount (for equivalency) of normal images from the dataset 
left_normal = df.loc[(df.A ==0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(300,random_state=42).values
right_normal = df.loc[(df.A ==0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(300,random_state=42).values
print(left_normal[:15])
print(right_normal[:15])


# In[11]:


#concating two arrays to make an array for age-related macular degeneration dataset (pic file names) and normal file names
armd = np.concatenate((left_armd,right_armd),axis=0)
normal = np.concatenate((left_normal,right_normal),axis=0)


# In[12]:


print(len(armd),len(normal))


# In[13]:


from tensorflow.keras.preprocessing.image import load_img,img_to_array
dataset_dir = "C:\\Users\\Downloads\\archive (7)\\preprocessed_images"
image_size=224 
labels = []
dataset = []
def create_dataset(image_category,label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir,img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)  
            image = cv2.resize(image,(image_size,image_size)) 
        except:
            continue 
        
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset


# In[14]:


dataset = create_dataset(armd,1)


# In[15]:


len(dataset)


# In[16]:


dataset = create_dataset(normal,0)


# In[17]:


len(dataset)


# In[18]:


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    if category== 0:
        label = "Normal"
    else:
        label = "ARMD"
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel(label)
plt.tight_layout() 


# In[19]:


x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
y = np.array([i[1] for i in dataset])


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[21]:


from tensorflow.keras.applications.vgg16 import VGG16
vgg16_model = VGG16(weights='imagenet', include_top=False)
vgg16_model.save_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[22]:


vgg16_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))


# In[23]:


from keras.applications.vgg16 import VGG16, preprocess_input
vgg16_weight_path = "C:\\Users\\Yukthamukhi\\Downloads\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
vgg = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=(224,224,3) l 
)


# In[24]:


for layer in vgg.layers:
    layer.trainable = False 


# In[25]:


from tensorflow.keras import Sequential
from keras import layers
from tensorflow.keras.layers import Flatten,Dense
model = Sequential() 
model.add(vgg)
model.add(Dense(256, activation='relu')) #1188 images, features are 224*224*3 for each image 
model.add(layers.Dropout(rate=0.5)) #deactivating 50% of the nodes from this layer 
model.add(Dense(128, activation='sigmoid'))
model.add(layers.Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(Flatten()) #linearizing the images. output of previous layer of size 224*224*3 is flattened to 1 dimension
model.add(Dense(1,activation="sigmoid")) 


# In[26]:


model.summary()


# In[27]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[28]:


history = model.fit(x_train,y_train,batch_size=32,epochs=15,validation_data=(x_test,y_test))


# In[29]:


loss,accuracy = model.evaluate(x_test,y_test)
print("loss:",loss)
print("Accuracy:",accuracy)


# In[30]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred1 = model.predict(x_test)
y_pred=np.argmax(y_pred1,axis=1)


# In[31]:


print(classification_report(y_test,y_pred))


# In[32]:


plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]
    
    if category== 0:
        label = "Normal"
    else:
        label = "Cataract"
        
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"
        
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 


# In[33]:


clf=model
p=0
q=1
prediction_train = clf.predict(x_train)
prediction_test = clf.predict(x_test)
for i in range(len(prediction_train)):
    if(prediction_train[i]>0.5):
        prediction_train[i]=1
    else:
        prediction_train[i]=0
        
for i in range(len(prediction_test)):
    if(prediction_test[i]>0.5):
        prediction_test[i]=1
    else:
        prediction_test[i]=0 


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
evaluation = pd.DataFrame({'Model': [],
                           'Accuracy(train)':[],
                           'Precision(train)':[],
                           'Recall(train)':[],
                           'F1_score(train)':[],
                           'Specificity(train)':[],
                           'Accuracy(test)':[],
                           'Precision(test)':[],
                           'Recalll(test)':[],
                           'F1_score(test)':[],
                           'Specificity(test)':[],
                          })
print(x_train.shape)
clf=model
acc_train=format(accuracy_score(prediction_train, y_train),'.3f')
precision_train=format(precision_score(y_train, prediction_train, average='binary'),'.3f')
recall_train=format(recall_score(y_train,prediction_train, average='binary'),'.3f')
f1_train=format(f1_score(y_train,prediction_train, average='binary'),'.3f')
tn, fp, fn, tp = confusion_matrix(prediction_train, y_train).ravel()
specificity = tn / (tn+fp)
specificity_train=format(specificity,'.3f')

acc_test=format(accuracy_score(prediction_test, y_test),'.3f')
precision_test=format(precision_score(y_test, prediction_test, average='binary'),'.3f')
recall_test=format(recall_score(y_test,prediction_test, average='binary'),'.3f')
f1_test=format(f1_score(y_test,prediction_test, average='binary'),'.3f')
tn, fp, fn, tp = confusion_matrix(prediction_test, y_test).ravel()
specificity = tn / (tn+fp)
specificity_test=format(specificity,'.3f')

r = evaluation.shape[0]
evaluation.loc[r] = ['Neural Network',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]
evaluation.sort_values(by = 'Accuracy(test)', ascending=False)


# In[38]:


pip install scikit-plot


# In[39]:


import scikitplot as skplt
p=y_train
q=y_test

y_train = pd.DataFrame(y_train)
y_train=y_train.replace([0,1], ["Negative","Positive"])


pred_train=prediction_train
pred_train=pd.DataFrame(pred_train)
pred_train=pred_train.replace([0,1], ["Negative","Positive"])


pred_test=prediction_test
y_test = pd.DataFrame(y_test)
y_test=y_test.replace([0,1], ["Negative","Positive"])
pred_test=pd.DataFrame(pred_test)

pred_test=pred_test.replace([0,1], ["Negative","Positive"])

skplt.metrics.plot_confusion_matrix(
    y_train, 
    pred_train,
    figsize=(7,4),
    title_fontsize='18',
    text_fontsize='16',
    title =' ',
    cmap='BuGn'
    )

skplt.metrics.plot_confusion_matrix(
    y_test, 
    pred_test,
    figsize=(7,4),
    title_fontsize='18',
    text_fontsize='16',
    title =' ',
    cmap='BuGn'
    )
y_train=p
y_test=q


# In[40]:


# n_samples, n_features = X.shape
y_score = prediction_test
n_classes = 2
#Create another array arr2 with size of arr1    
roc_y_test = prediction_test
    
y_score[50]=2
roc_y_test[50]=2
y_test
from sklearn.preprocessing import label_binarize
y_score = label_binarize(y_score, classes=[1, 0])
roc_y_test = label_binarize(roc_y_test, classes=[1, 0])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(roc_y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(roc_y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




# First aggregate all false positive rates
lw = 2
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CNN')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




