import random
import json
import torch
import speech_recognition as sr
import pyttsx3
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


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
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # listens to audio
                audio2 = r.listen(source2)

                # using google to recognize audio
                MyText = r.recognize_google(audio2)
                return MyText

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unkbown error occured")




bot_name = "Medbot"


while True:
    sentence = speech_to_text()
    print("Let's chat! Say 'quit' to exit")
    print(f"You: {sentence}")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.70:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}\n")
                speckText(response)
                
    else:
        print(f"{bot_name}: I do not understand...")
        speckText("I do not understand")
