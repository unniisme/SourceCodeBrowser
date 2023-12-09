
class Model:

    def __init__(self):
        pass

    def predict(self, sentance : str):
        return "#Summary :" + sentance

def get_model(name):
    if name == "gpt":
        print("Loading GPT API model")
        from model.gpt__api import GPTAPIModel
        return GPTAPIModel()
    
    if name == "T5":
        print("Loading T5 model")
        from model.T5 import T5Model
        return T5Model()
    