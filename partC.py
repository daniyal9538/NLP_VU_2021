import csv
import datetime
import os
import random
import re
import time
import warnings
import wget

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup, logging

print('Setting things up...')

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

print('Loading the dataset...')

url = 'https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskB.txt'
if not os.path.exists('./SemEval2018-T3-train-taskB.txt'):
    wget.download(url, './SemEval2018-T3-train-taskB.txt')

df = pd.read_csv('./SemEval2018-T3-train-taskB.txt', delimiter='\t', index_col=0,
                 header=0, names=['label', 'tweet'], quoting=csv.QUOTE_NONE,
                 error_bad_lines=False)[['tweet', 'label']]

print('Number of training tweets: {:,}\n'.format(df.shape[0]))
print(df.sample(10))
print(df.loc[df.label == 0].sample(5)[['tweet', 'label']])

tweets = df.tweet.values
labels = df.label.values

print('Preprocessing tweets...')


def process_tweet(t):
    t = re.sub(r'^https?://.*[\r\n]*', '', t, flags=re.MULTILINE)
    t = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)',
               '', t, flags=re.MULTILINE)
    t = re.sub(r'(:.+:)', '', t, flags=re.MULTILINE)
    return t


tweets = [process_tweet(t) for t in tweets]

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

print(' Original: ', tweets[0])
print('Tokenized: ', tokenizer.tokenize(tweets[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tweets[0])))

input_ids = [tokenizer.encode(t, add_special_tokens=True) for t in tweets]

print('Original: ', tweets[0])
print('Token IDs:', input_ids[0])

print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

max_length = max([len(t) for t in tweets])
print(f'Max length: {max_length}')

input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids],
                         batch_first=True, padding_value=0)

print('\nDone.')

attention_masks = []

for t in input_ids:
    att_mask = [int(token_id > 0) for token_id in t]
    attention_masks.append(att_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                    random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                       random_state=2018, test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

print('Loading model...')

model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(set(labels)),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.cuda()

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_values = []

for epoch_i in range(0, epochs):
    print('')
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]

        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()

    avg_train_loss = total_loss / len(train_dataloader)

    loss_values.append(avg_train_loss)

    print('')
    print('  Average training loss: {0:.2f}'.format(avg_train_loss))
    print('  Training epcoh took: {:}'.format(format_time(time.time() - t0)))

    print('')
    print('Running Validation...')

    t0 = time.time()
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print('  Accuracy: {0:.2f}'.format(eval_accuracy / nb_eval_steps))
    print('  Validation took: {:}'.format(format_time(time.time() - t0)))

print('')
print('Training complete!')

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 6)

# Plot the learning curve.
plt.plot(loss_values, 'b-o')

# Label the plot.
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()

url = 'https://raw.githubusercontent.com/dearden/SemEval2018-Irony/master/SemEval2018-T3_gold_test_taskB_emoji.txt'
if not os.path.exists('./SemEval2018-T3_gold_test_taskB_emoji.txt'):
    wget.download(url, './SemEval2018-T3_gold_test_taskB_emoji.txt')


df = pd.read_csv('./SemEval2018-T3_gold_test_taskB_emoji.txt', delimiter='\t', index_col=0,
                 header=0, names=['label', 'tweet'], quoting=csv.QUOTE_NONE,
                 error_bad_lines=False)[['tweet', 'label']]

print('Number of test tweets: {:,}\n'.format(df.shape[0]))

tweets = df.tweet.values
labels = df.label.values

input_ids = [tokenizer.encode(t, add_special_tokens=True) for t in tweets]
input_ids = pad_sequence([torch.tensor(input_id) for input_id in input_ids], batch_first=True, padding_value=0)

attention_masks = []

for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

batch_size = 32

prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
print('Predicting labels for {:,} test tweets...'.format(len(prediction_inputs)))

model.eval()

predictions, true_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')

print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))

label_weights = {}
for label in labels:
    label_weights[label] = len(df[df['label'] == label]) / len(df)

matthews_set = []
f1_score_set = []
confusion_matrix_set = []

print('Calculating Matthews Corr. Coef. for each batch...')

for i in range(len(true_labels)):
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    sample_weight = [label_weights[label] for label in true_labels[i]]

    matthews_set.append(matthews_corrcoef(true_labels[i], pred_labels_i))
    f1_score_set.append(f1_score(true_labels[i], pred_labels_i,
                                 sample_weight=sample_weight,
                                 labels=list(label_weights.keys()),
                                 average='weighted'))
    confusion_matrix_set.append(confusion_matrix(true_labels[i], pred_labels_i,
                                                 sample_weight=sample_weight,
                                                 labels=list(label_weights.keys())))

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

flat_df_columns = pd.Index(["predictions", "true_labels"], name="columns")
flat_df = pd.DataFrame(data=np.array([flat_predictions, flat_true_labels]).T, columns=flat_df_columns)
flat_df.to_csv("predictions-true_labels.csv", index=False)

for i in [0, 1, 2, 3]:
    _df = flat_df[flat_df['true_labels'] == i]
    print("{:.2f}".format(accuracy_score(_df['true_labels'], _df['predictions'])))

precision = precision_score(flat_true_labels, flat_predictions, average=None, labels=[0, 1, 2, 3])
precision_macro = precision_score(flat_true_labels, flat_predictions, average='macro')
precision_weighted = precision_score(flat_true_labels, flat_predictions, average='weighted')

print(["{:.2f}".format(p) for p in precision])
print("{:.2f}".format(precision_macro))
print("{:.2f}".format(precision_weighted))

recall = recall_score(flat_true_labels, flat_predictions, average=None, labels=[0, 1, 2, 3])
recall_macro = recall_score(flat_true_labels, flat_predictions, average='macro')
recall_weighted = recall_score(flat_true_labels, flat_predictions, average='weighted')

print(["{:.2f}".format(r) for r in recall])
print("{:.2f}".format(recall_macro))
print("{:.2f}".format(recall_weighted))

f1 = f1_score(flat_true_labels, flat_predictions, average=None, labels=[0, 1, 2, 3])
f1_macro = f1_score(flat_true_labels, flat_predictions, average='macro')
f1_weighted = f1_score(flat_true_labels, flat_predictions, average='weighted')

print(["{:.2f}".format(f) for f in f1])
print("{:.2f}".format(f1_macro))
print("{:.2f}".format(f1_weighted))

cm = confusion_matrix(flat_true_labels, flat_predictions, labels=[0, 1, 2, 3])
print(cm)

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

labels = list(range(4))

fig, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="w")

ax.xaxis.tick_top()

ax.set_title("Confusion Matrix: Finetuned BERT")
ax.set_xlabel("Predicted Class")
ax.xaxis.set_label_position('top')
ax.set_ylabel("Gold Class")

fig.tight_layout()
plt.show()

fig.savefig("cm.png")
