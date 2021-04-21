import tensorflow as tf
import wget
import os
import pandas as pd
import csv
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import random
import time
import datetime
import torch
from transformers import get_linear_schedule_with_warmup
import wandb
# Get the GPU device name.


def downloadData():
    url = 'https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/543ae83b0b093ab7ff4f2891ab7bccf6721d7e6d/datasets/train/SemEval2018-T3-train-taskB.txt'

    # Download the file (if we haven't already)
    if not os.path.exists('./SemEval2018-T3-train-taskB.txt'):
        wget.download(url, './SemEval2018-T3-train-taskB.txt')

def tokenize(df):
    sentences = df.tweet.values


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    return input_ids


def padSentences(input_ids):
    MAX_LEN = 131
    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids_padded = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids_padded:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return input_ids_padded, attention_masks

def getDataLoaders(train_inputs, train_masks, train_labels, validation_inputs, validation_masks, validation_labels, batch_size):


    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def trainModel(train_inputs, validation_inputs, train_labels, validation_labels,train_masks, validation_masks, device):
    wandb.init(project="NLP-A1", reinit=True)

    config = wandb.config

    batch_size = 32
    train_dataloader, validation_dataloader = getDataLoaders(train_inputs, train_masks, train_labels, validation_inputs,
                                                             validation_masks, validation_labels, batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4, output_attentions=False,
                                                          output_hidden_states=False)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    epochs = config.epochs

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    wandb.watch(model, log='all', log_freq=20)


    for epoch_i in range(0, config.epochs):
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        wandb.log({'Train loss': avg_train_loss})

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions = []
        true_labels = []
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
            true_labels.append(b_labels.detach().cpu().numpy())
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1


        predictions = np.argmax(np.vstack(predictions), axis=1).flatten()

        macro_f1 = f1_score(np.hstack(true_labels), predictions, zero_division=1, average='macro')
        print("Macro F1:", macro_f1)
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        wandb.log({'Macro f1': macro_f1, 'Accuracy': (eval_accuracy / nb_eval_steps)})


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    downloadData()
    # df = pd.read_csv("./SemEval2018-T3-train-taskB.txt", delimiter='\t', quoting=csv.QUOTE_NONE, error_bad_lines=False)
    df = pd.read_csv("./SemEval2018-T3-train-taskB.txt", delimiter='\t', index_col=0, header=0, names=["label", "tweet"])[["tweet", "label"]]
    labels = df.label.values

    input_ids = tokenize(df)
    input_ids_padded, attention_masks = padSentences(input_ids)

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_padded, labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                           random_state=2018, test_size=0.1)


    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    print(len(validation_labels))

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    trainModel(train_inputs, validation_inputs, train_labels, validation_labels,train_masks, validation_masks, device)


if __name__ == '__main__':
    main()








