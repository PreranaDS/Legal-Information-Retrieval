import speech_recognition as sp
speech=sp.Microphone()
with  speech as source:
    print("Speak Anything")
    audio=speech.listen(source)
    text=speech.recognize_google(audio)
    print("you said {}".format(text))