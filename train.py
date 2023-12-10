import torch
from transformers import Trainer, TrainingArguments

from model.T5 import T5Model

import json
import logging
import sys
import os

FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s()] %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stdout, encoding='utf-8', level=logging.DEBUG)

with open(f"./config/{sys.argv[1]}.json") as f:
    CONFIG = json.load(f)


def TokenizePair(data_pair, tokenizer):

    input_text = f"{data_pair['code']} </s>"
    target_text = f"{data_pair['docstring']} </s>"

    encoded = tokenizer(
        input_text,
        # pad_to_max_length=True,
        padding='max_length',
        truncation='longest_first',
        max_length=128,
        return_tensors='pt'
    )

    decoded = tokenizer(
        target_text,
        # pad_to_max_length=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    output_ids = decoded.input_ids
    
    return {    
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels' : output_ids
    }


logging.info("Loading model")
t5model = T5Model()

logging.info("Defining dataset")
train_dataset = []
jsons = [f"{CONFIG["DataStorePath"]}/{f}" for f in os.listdir(CONFIG["DataStorePath"])]
for js in jsons:
    if js.endswith(".json"):
        with open(js) as f:
            data = json.load(f)
            for pair in data:
                data_pair = TokenizePair(pair, t5model.tokenizer)
                data_pair["file"] = js
                train_dataset.append(data_pair)

# train_dataset = [TokenizePair(pair, t5model.tokenizer) for pair in algorithms]
logging.info(f"Dataset size = {len(train_dataset)}")


model = t5model.model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

logging.info("Training")

# Fine-tuning the model
num_epochs = 10
# try:
for epoch in range(num_epochs):
    for batch in train_dataset:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # print(f"File {batch["file"]} Loss : {loss.item()}", end="\r")
        logging.debug(f"File {batch["file"]} Loss : {loss.item()}")

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")
    logging.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")
# except:
#     logging.error("Exiting with error")

#     logging.info("Saving")
#     import time
#     # Save the model if failed
#     timestamp = time.time()
#     model.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model.{timestamp}")
#     t5model.tokenizer.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model.{timestamp}")
    
#     quit()

logging.info("Saving")
# Save the model after training
model.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model")
t5model.tokenizer.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model")