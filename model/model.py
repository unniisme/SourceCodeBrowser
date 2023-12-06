

class DummyModel:

    def __init__(self):
        pass

    def predict(self, sentance : str):
        return "#Yassified\n " + sentance

def get_model():
    return DummyModel()