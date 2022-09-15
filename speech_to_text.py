import speech_recognition as sr
import pyttsx3


r = sr.Recognizer()

def speckText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


while(1):
    try:
        # use microphone
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.1)

            # listens to audio
            audio2 = r.listen(source2)

            # using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("Did you say: " + MyText)
            speckText(MyText)
            if "goodbye" in MyText:
                break
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unkbown error occured")