import random
import json
import torch
import speech_recognition as sr
import pyttsx3
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from flask import Flask, request, jsonify


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
r = sr.Recognizer()

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()



def speckText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def speech_to_text():
    while(1):
        try:
            # use microphone
            with sr.Microphone() as source2:
                print("Listening...")
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # listens to audio
                audio2 = r.listen(source2)

                # using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print(MyText)
                return MyText

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unkbown error occured")




bot_name = "Medbot"
print("Let's chat! Say 'quit' to exit")

def sen_token(human_input):
    sentence = tokenize(human_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.80:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return (response)
                # speckText(response)
                
    else:
        return ("I do not understand...")
        


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = str(request.data)
        input_data = input_data.removeprefix("b'")
        input_data = input_data.removesuffix("'")
        # print(input_data)
        if input_data is None:
            return jsonify({'error': 'no input'})

    try:
        sentence = input_data
        response = sen_token(sentence)
        return response
    except:
        return jsonify({'error': 'error during prediction'})


# while True:
#     respones = sen_token(speech_to_text())
#     print(respones)
#     speckText(respones)