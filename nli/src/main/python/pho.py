#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Generator


# In[3]:


# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = "cpu"


# In[2]:


from os.path import expanduser
home = expanduser("~")
path = home + "/vlp/dat/nli/XNLI-1.0/vi.tok.jsonl"
print(path)


# In[3]:


class VietnameseXNLI(Dataset):
    def __init__(self, jsonlPath):
        self.X = []
        self.y = []
        with open(jsonlPath) as f:
            for line in f:
                sample = json.loads(line)
                self.X.append(sample["sentence1_tokenized"] +
                              " </s> " + sample["sentence2_tokenized"])
                self.y.append(sample["gold_label"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# In[4]:


dataset = VietnameseXNLI(path)
N = int(0.8*len(dataset))
training, test = random_split(dataset, [N, len(dataset)-N], generator=Generator().manual_seed(12345))


# In[117]:


batch_size=32


# In[5]:


# for testing only
train_loader = DataLoader(training, batch_size=batch_size)
test_loader = DataLoader(test, batch_size=batch_size)


# In[4]:


next(iter(train_loader))


# In[8]:


from transformers import AutoModel, AutoTokenizer


# In[9]:


bert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


# In[10]:


tokenizer


# In[11]:


bert


# In[44]:


label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}


# In[37]:


import transformers
from transformers import (
    BertPreTrainedModel,
    RobertaConfig
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)
import torchtext
from torch.nn import CrossEntropyLoss


# In[39]:


class RobertaPhoBERTClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaPhoBERTClassifier, self).__init__(config)
        self.num_labels = 3
        self.model = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs  # (loss,) logits, (hidden_states, (attentions))


# In[40]:


# https://pchanda.github.io/Roberta-FineTuning-for-Classification/
config_class = RobertaConfig
model_class = RobertaPhoBERTClassifier
config = config_class.from_pretrained("vinai/phobert-base", num_labels=3)
model = model_class.from_pretrained("vinai/phobert-base", config=config)

model


# In[93]:


class ClassificationDataset(Dataset):
    def __init__(self, data, tokenizer):
        xs, ys = data
        # this is like calling tokenizer.encode() but has paddings
        self.examples = tokenizer(text=xs, text_pair=None, truncation=True, padding='max_length',
                                  max_length=tokenizer.model_max_length, return_tensors='pt')
        self.labels = torch.tensor([label_dict[y] for y in ys], dtype=torch.long)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]


# In[78]:


# each data is a pair of (xs, ys)
training_data = ([x[0] for x in training], [x[1] for x in training])
test_data = ([x[0] for x in test], [x[1] for x in test])


# In[80]:


training_ds = ClassificationDataset(training_data, tokenizer)
test_ds = ClassificationDataset(test_data, tokenizer)


# In[96]:


# create data loader
train_loader = DataLoader(training_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)


# In[98]:


first_test_batch = next(iter(test_loader))


# In[101]:


# xs in the first batch
first_test_batch[0]


# In[104]:


# ys in the first batch
first_test_batch[1]


# In[106]:


def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(DEVICE) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(DEVICE)
    return inputs


# In[107]:


get_inputs_dict(first_test_batch)


# In[109]:


batch = get_inputs_dict(first_test_batch)
first_input_ids = batch['input_ids'].to(DEVICE)
first_attention_mask = batch['attention_mask'].to(DEVICE)
first_labels = batch['labels'].to(DEVICE)


# In[110]:


print(batch)


# In[ ]:


model.to(DEVICE)


# In[112]:


# for testing
model(first_input_ids, first_attention_mask, first_labels)


# In[113]:


import math
from transformers.optimization import (
    AdamW, 
    get_linear_schedule_with_warmup
)

from scipy.special import softmax
from torch.nn import CrossEntropyLoss

from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
    average_precision_score,
)


# In[119]:


warmup_ratio = 0.06
weight_decay=0.0
gradient_accumulation_steps = 1
num_train_epochs = 15
learning_rate = 1e-05
adam_epsilon = 1e-08
t_total = len(train_loader) // gradient_accumulation_steps * num_train_epochs
optimizer_grouped_parameters = []
custom_parameter_names = set()
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters.extend(
    [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
)

warmup_steps = math.ceil(t_total * warmup_ratio)
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)


# In[118]:


import numpy as np

def compute_metrics(preds, model_outputs, labels, eval_examples=None, multi_label=False):
    assert len(preds) == len(labels)
    mismatched = labels != preds
    accuracy = sum(mismatched)/len(labels)
    # wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
    # mcc = matthews_corrcoef(labels, preds)
    # tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    # scores = np.array([softmax(element)[1] for element in model_outputs])
    # fpr, tpr, thresholds = roc_curve(labels, scores)
    # auroc = auc(fpr, tpr)
    # auprc = average_precision_score(labels, scores)
    # return (
    #     {
    #         **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "auroc": auroc, "auprc": auprc},
    #     },
    #     wrong,
    # )
    return accuracy
    
def print_confusion_matrix(result):
    print('confusion matrix:')
    print('            predicted    ')
    print('          0     |     1')
    print('    ----------------------')
    print('   0 | ',format(result['tn'],'5d'),' | ',format(result['fp'],'5d'))
    print('gt -----------------------')
    print('   1 | ',format(result['fn'],'5d'),' | ',format(result['tp'],'5d'))
    print('---------------------------------------------------')


# In[121]:


model.zero_grad()

for epoch in range(num_train_epochs):
    model.train()
    epoch_loss = []
    
    for batch in train_loader:
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        epoch_loss.append(loss.item())
        
    # evaluate model on test data at the end of the epoch.
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_loader)
    preds = np.empty((len(test_ds), 3))
    out_label_ids = np.empty((len(test_ds)))
    model.eval()
    
    for i, test_batch in enumerate(test_loader):
        with torch.no_grad():
            test_batch = get_inputs_dict(test_batch)
            input_ids = test_batch['input_ids'].to(DEVICE)
            attention_mask = test_batch['attention_mask'].to(DEVICE)
            labels = test_batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()
            
        nb_eval_steps += 1
        start_index = batch_size * i
        end_index = start_index + batch_size if i != (n_batches - 1) else len(test_ds)
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()
        
    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)
    # result, wrong = compute_metrics(preds, model_outputs, out_label_ids, test_data)
    accuracy = compute_metrics(preds, model_outputs, out_label_ids, test_data)
    
    print('epoch ', epoch, 'Training avg loss', np.mean(epoch_loss))
    print('epoch ', epoch, 'Testing  avg loss', eval_loss)
    print('accuracy = ', accuracy)
    # print_confusion_matrix(result)
    print(model_outputs[0:4])
    print('---------------------------------------------------\n')

