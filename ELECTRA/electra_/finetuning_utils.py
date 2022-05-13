from transformers import ElectraTokenizer, ElectraForSequenceClassification

def model_init():
    model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator",num_labels=2) # modify num_labels on demand
    return model

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
    accurcay = accuracy_score(labels,preds)
    NMI = normalized_mutual_info_score(labels,preds)
    ARI = adjusted_rand_score(labels,preds)
    return {'eval_accuracy':accurcay,'eval_NMI':NMI,'eval_ARI':ARI}

