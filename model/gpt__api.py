from openai import OpenAI


class GPTAPIModel:

    def __init__(self):
        self.client = OpenAI()
        self.systemPrompt = "You are a code summarizer. You summarize all code that a user inputs."
    
    def predict(self, s : str):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.systemPrompt},
                {"role": "user", "content": s}
            ]
        )

        return completion.choices[0].message.content
