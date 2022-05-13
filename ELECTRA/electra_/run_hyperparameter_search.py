import argparse
import dataset
import data_utils
from dataset import CreateDataset
import finetuning_utils
from finetuning_utils import model_init 
from finetuning_utils import compute_metrics
import json
import pandas as pd
import pprint as pp

from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a ELECTRA model on the agNews/yahoo_answers dataset."
)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--class_num', type=int)
args = parser.parse_args()

train_data=pd.read_csv(f"{args.data_dir}_train.csv",header=0,index_col=0,names=['classid','text'])
test_data=pd.read_csv(f"{args.data_dir}_test.csv",header=0,index_col=0,names=['classid','text'])
train_data['classid'] = train_data['classid'].map(dict(zip(range(1,args.class_num+1), range(args.class_num))))
test_data['classid'] = test_data['classid'].map(dict(zip(range(1,args.class_num+1), range(args.class_num))))
train_data['text'] = train_data['text'].values.astype('str')
test_data['text'] = test_data['text'].values.astype('str')

tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

# split train_data into training and val data
training_data = train_data.sample(frac=0.8, random_state=0)
val_data = train_data.drop(training_data.index)

train_data_electra = CreateDataset(training_data, tokenizer)
val_data_electra = CreateDataset(val_data, tokenizer)
test_data_electra = CreateDataset(test_data, tokenizer)

training_args = TrainingArguments(
    output_dir="/scratch_tmp/yg2483/models",
    do_train=True,
    do_eval=True,
    no_cuda=1 <= 0, #
    per_device_train_batch_size=5, 
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=1000,  # save parameters
    logging_first_step=True,
    save_steps=1500, # save models
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
    logging_dir="/scratch_tmp/yg2483/logs",
    learning_rate=1e-5, # config
    weight_decay=0.01,
)

training_args._n_gpu = 1 #

trainer = Trainer(args = training_args,
                  train_dataset=train_data_electra,
                  eval_dataset=val_data_electra,
                  tokenizer=tokenizer,
                  model_init = model_init,
                  compute_metrics = compute_metrics,)

tune_config = {"learning_rate": tune.uniform(1e-5, 5e-5)} 

reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
        },
        metric_columns=[
            "eval_accuracy", "eval_NMI", "eval_ARI"
        ])

best_results = trainer.hyperparameter_search(
    hp_space = lambda _:tune_config,
    backend = 'ray',
    compute_objective = lambda metrics: metrics["eval_ARI"],
    mode = 'max',
    search_alg = BasicVariantGenerator(),
    n_trials=3, 
    progress_reporter=reporter,
    resources_per_trial={"cpu":8, "gpu":1},
)

pp.pprint(best_results)


