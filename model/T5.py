from transformers import RobertaTokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
import logging

pretrained = lambda lang : f"pretrained/summarize_{lang}_codet5_base.bin"

class T5Model:

    def __init__(self, lang = None):
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

        if lang != None:
            try:
                self.model.load_state_dict(torch.load(pretrained(lang)))
                logging.info(f"Loaded model for {lang}")
            except Exception as e:
                logging.error(e)

    def predict(self,text):

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        generated_ids = self.model.generate(input_ids, max_length=512)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
class T5Model_Pretrained(T5Model):

    def __init__(self, model_path):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
    

class T5_plus(T5Model):

    def __init__(self):
        checkpoint = "Salesforce/codet5p-220m-py"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)