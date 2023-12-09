import torch

from model.T5 import T5Model

import json
import logging
import sys
import os

FORMAT = "[%(levelname)-8s][%(asctime)s][%(filename)s:%(lineno)s - %(funcName)13s()] %(message)s"
logging.basicConfig(format=FORMAT, filename="log/training.log", encoding='utf-8', level=logging.DEBUG)

with open(f"./config/{sys.argv[1]}.json") as f:
    CONFIG = json.load(f)


def TokenizePair(data_pair, tokenizer):

    input_text = f"{data_pair['code']} </s>"
    target_text = f"{data_pair['docstring']} </s>"

    encoded = tokenizer(
        input_text,
        target_text,
        # pad_to_max_length=True,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoded.input_ids.flatten()
    attention_mask = encoded.attention_mask.flatten()
    
    return {    
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels' : input_ids.clone()
    }

# Loading dataset
class CustomIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def __iter__(self):
        for file_path in self.file_paths:
            logging.debug(f"loading {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                for data_pair in data:
                    tokenizedPair = TokenizePair(data_pair, self.tokenizer)
                    tokenizedPair["file"] = file_path
                    yield tokenizedPair 


logging.info("Loading model")
t5model = T5Model()

logging.info("Defining dataset")
jsons = [f"{CONFIG["DataStorePath"]}/{f}" for f in os.listdir(CONFIG["DataStorePath"])]
train_dataset = CustomIterableDataset(jsons, t5model.tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)


model = t5model.model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

logging.info("Training")
# Fine-tuning the model
num_epochs = 10
try:
    for epoch in range(num_epochs):
        for batch in train_loader:
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

            print(f"File {batch["file"]} Loss : {loss.item()}", end="\r")
            logging.debug(f"File {batch["file"]} Loss : {loss.item()}")

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")
except:
    logging.error("Exiting with error")

    logging.info("Saving")
    import time
    # Save the model if failed
    timestamp = time.time()
    model.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model.{timestamp}")
    t5model.tokenizer.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model.{timestamp}")
    
    quit()

logging.info("Saving")
# Save the model after training
model.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model")
t5model.tokenizer.save_pretrained(f"{CONFIG['ModelStorePath']}/fine_tuned_model")