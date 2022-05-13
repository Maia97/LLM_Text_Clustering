import argparse
import torch
import numpy as np
import pandas as pd
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="Evaluate the best model with 3 metrics: ACC, NMI, ARI."
)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--class_num', type=int)
parser.add_argument('--best_trial', type=str)
args = parser.parse_args()

train_data=pd.read_csv(f"{args.data_dir}_train.csv",header=0,index_col=0,names=['classid','text'])
test_data=pd.read_csv(f"{args.data_dir}_test.csv",header=0,index_col=0,names=['classid','text'])
train_data['classid'] = train_data['classid'].map(dict(zip(range(1,args.class_num+1), range(args.class_num))))  
test_data['classid'] = test_data['classid'].map(dict(zip(range(1,args.class_num+1), range(args.class_num)))) 
train_data['text'] = train_data['text'].values.astype('str')
test_data['text'] = test_data['text'].values.astype('str')

tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

model_finetune = ElectraForSequenceClassification.from_pretrained(f"/scratch_tmp/yg2483/models/run-{args.best_trial}/checkpoint-1500",num_labels=args.class_num, output_hidden_states=True)
model_pretrain = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator",num_labels=args.class_num,output_hidden_states=True)

true_label = test_data['classid']

def text_representation(dataframe,model,tokenizer):
    representation = []
    for i in range(len(dataframe)):
        text = dataframe['text'].iloc[i]
        inputs = tokenizer(text, max_length=64,padding="max_length",truncation=True,return_tensors="pt")
        outputs = model(**inputs)
        length = np.array(inputs['attention_mask'][0]).sum()
        encoding = outputs.hidden_states[-1][0].detach().numpy()[:length,:]
        encoding = list(encoding.mean(axis=0))
        representation.append(encoding)
    return np.array(representation)

pretrain_train = text_representation(train_data,model_pretrain,tokenizer)
pretrain_test = text_representation(test_data,model_pretrain,tokenizer)
finetune_train = text_representation(train_data,model_finetune,tokenizer)
finetune_test = text_representation(test_data,model_finetune,tokenizer)

def three_metrics(true_label,preds):  
    from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
    ACC = round(accuracy_score(true_label,preds),3)
    if ACC<=1/len(true_label.unique()): #说明：聚类的label，和真实的label没对上（改好了，不需要调整）
        keys = list(pd.value_counts(preds).index)
        values = list(pd.value_counts(true_label).index)
        dic = dict(zip(keys, values))
        preds = pd.Series(preds).map(dic)
    NMI = round(normalized_mutual_info_score(true_label,preds),3)
    ARI = round(adjusted_rand_score(true_label,preds),3)
    ACC = round(accuracy_score(true_label,preds),3)
    return {'ACC':ACC,'NMI':NMI,'ARI':ARI}

#K-Means
from sklearn.cluster import KMeans
print('K-Means')
clustering_model = KMeans(n_clusters = args.class_num, 
                          init = 'k-means++',
                          max_iter = 300, n_init = 10,random_state=123)
clustering_model.fit(pretrain_train)
pretrain_KMeans = clustering_model.predict(pretrain_test)
clustering_model.fit(finetune_train)
finetune_KMeans = clustering_model.predict(finetune_test)
print('pretrained text representation:',three_metrics(true_label,pretrain_KMeans))
print('finetuned text representation:',three_metrics(true_label,finetune_KMeans))

#FCM
from fcmeans import FCM
print('FCM')
fcm = FCM(n_clusters=args.class_num)
fcm.fit(np.array(pretrain_train))
pretrain_FCM = fcm.predict(pretrain_test)
fcm.fit(np.array(finetune_train))
finetune_FCM = fcm.predict(finetune_test)
print('pretrained text representation:',three_metrics(true_label,pretrain_FCM))
print('finetuned text representation:',three_metrics(true_label,finetune_FCM))

#LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
print('LDA')
scaler = MinMaxScaler()
pretrain_train2 = scaler.fit_transform(pretrain_train)
pretrain_test2 = scaler.fit_transform(pretrain_test)
finetune_train2 = scaler.fit_transform(finetune_train)
finetune_test2 = scaler.fit_transform(finetune_test)

lda = LatentDirichletAllocation(n_components=args.class_num, random_state=456)
lda.fit(pretrain_train2)
doc_topic_dist_unnormalized = np.matrix(lda.transform(pretrain_test2))
doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
pretrain_LDA = list(np.array(doc_topic_dist.argmax(axis=1)).T[0])
lda.fit(finetune_train2)
doc_topic_dist_unnormalized = np.matrix(lda.transform(finetune_test2))
doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
finetune_LDA = list(np.array(doc_topic_dist.argmax(axis=1)).T[0])
print('pretrained text representation:',three_metrics(true_label,pretrain_LDA))
print('finetuned text representation:',three_metrics(true_label,finetune_LDA))
