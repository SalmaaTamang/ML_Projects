# %%
import pandas as pd


# %%
train = pd.read_csv('cleaned_train.csv', encoding = 'utf-8')
test = pd.read_csv('cleaned_test.csv', encoding = 'utf-8')

# %%
dataset = pd.DataFrame.merge(train,test,how='outer')
dataset.shape

# %%
dataset.isnull().sum()

# %%
dataset = dataset.dropna()

# %%
dataset.isnull().sum()

# %%
dataset.label.value_counts()

# %%
one_hot =pd.get_dummies(dataset.label, prefix='label')
one_hot

# %%
one_hot = one_hot.astype(int)
one_hot

# %%
dataset['label0'] = one_hot['label_0']
dataset['label1'] = one_hot['label_1']
dataset['label2'] = one_hot['label_2']
dataset

# %%
dataset= dataset.drop(columns=['label'])
dataset

# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size =0.2, random_state=100)

# %%
train.shape


# %%
test.shape

# %%
train = train.reset_index()
train.head()

# %%
test = test.reset_index()
test.head()

# %%
! pip install transformers -q

# %%
traget_cols = ['label0','label1','label2']
traget_cols

# %%
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import  AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

# %%
#initalizing which tokenizer to be used
tokenizer = AutoTokenizer.from_pretrained('Sakonii/deberta-base-nepali')

# %%
a = '''Hello world'''
tout = tokenizer(a)
tout

# %%
tout1 = tokenizer(a, max_length = 15, padding= 'max_length', truncation=True)
tout1

# %%
tout1 = tokenizer(a, max_length = 15, padding= 'max_length', truncation=True, add_special_tokens=True, return_token_type_ids=True)
tout1

# %%
! pip install tf-keras

# %%
from transformers import  AutoModelForMaskedLM

muril = AutoModelForMaskedLM.from_pretrained('google/muril-base-cased', from_tf=True)

input_ids = torch.tensor(tout1["input_ids"]).unsqueeze(0)           
attention_mask = torch.tensor(tout1["attention_mask"]).unsqueeze(0) 
token_type_ids = torch.tensor(tout1["token_type_ids"]).unsqueeze(0) 

outputs = muril(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
outputs


# %%
outputs[0]


# %%
max_len = 192

# %%
class BertDataset(Dataset):
  def __init__(self, dataset,tokenizer, max_len):
    self.dataset = dataset
    self.max_len  = max_len
    self.text = dataset.text
    self.tokenizer = tokenizer
    self.targets = dataset[traget_cols].values
  
  def __len__(self):
    return(len(self.dataset))

  
  def __getitem__(self, index):
    text = self.text[index]
    tokens = self.tokenizer(text, max_length = self.max_len, padding = 'max_length', truncation = True, add_special_tokens=True, return_token_type_ids=True)
    ids = tokens['input_ids']
    mask = tokens['attention_mask']
    token_type_ids = tokens['token_type_ids']

    return {
       'ids':torch.tensor(ids, dtype = torch.long),
       'mask':torch.tensor(mask, dtype = torch.long),
       'token_type_ids':torch.tensor(token_type_ids, dtype = torch.long),
       'targets':torch.tensor(self.targets[index], dtype = torch.float)
           } 


# %%
train_dataset = BertDataset(train, tokenizer,max_len)
test_dataset = BertDataset(test, tokenizer,max_len)

# %%
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=False, pin_memory=True)
     

# %%
from transformers import AutoModelForMaskedLM

class BertClass(torch.nn.Module):
  def __init__(self):
    super(BertClass, self).__init__()
    self.muril = AutoModelForMaskedLM.from_pretrained('google/muril-base-cased',from_tf = True)
    self.dense1 = torch.nn.Linear(197285,512)
    self.dropout1 = torch.nn.Dropout(0.5)
    self.dense2 = torch.nn.Linear(512,256)
    self.dropout2 = torch.nn.Dropout(0.5)
    self.fc = torch.nn.Linear(256,3)

  def forward (self, ids, mask, token_type_ids):
    outputs = self.muril(ids, attention_mask=mask, token_type_ids=token_type_ids)
    features = outputs[0]
    x = self.dense1(features)
    x = self.dropout1(x)
    x = self.dense2(x)
    x = self.dropout2(x)
    outputs = self.fc(x)

    return outputs


# %%
a = BertClass()
out = a(input_ids, attention_mask, token_type_ids)
out


# %%
out.shape

# %%
for _, batch in enumerate(train_loader):
    print("Batch (dictionary):", batch.keys() if isinstance(batch, dict) else "Not a dictionary")
    break


# %%
targets = batch["targets"] 
targets.shape



# %%
id = batch["ids"]
id.shape

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")
        

# %%
  
model = BertClass()
model.to(device)

# %%
torch.cuda.empty_cache()

# %%
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# %%

optimizer = AdamW(params =  model.parameters(), lr=5e-5, weight_decay=1e-5)
     


# %%
scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=50, num_training_steps=100)

# %%
import numpy as np
from sklearn import metrics
def validation():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)  
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


# %%

def train(epoch):
    model.train()
    for _,data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
    outputs, targets = validation()
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print('\n')
     

# %%
for epoch in range(3):
  train(epoch)

# %%
outputs, targets = validation()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
f1_score_weighted = metrics.f1_score(targets, outputs, average='weighted')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
print(f"F1 Score (Weighted) = {f1_score_weighted}")  


