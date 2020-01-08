from flask import Flask, request
app = Flask(__name__)

from .brain import create_model, predict
from .data import get_tokenizer

print("Loading model")

tokenizer = get_tokenizer()
VOCAB_SIZE = tokenizer.vocab_size + 2
WEIGHTS = "../jarvis.h5"
model = create_model(VOCAB_SIZE, WEIGHTS)

print("Model Loaded!!")


@app.route('/')
def hello_world():
    sentence = request.args.get("sentence", "World")
    return predict(model,sentence,tokenizer)



if __name__ == '__main__':
    app.run()

