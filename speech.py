import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
def voiceconversion():
    r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

    with sr.Microphone() as source:
        print("Talk")
        audio_text = r.listen(source,2.0)
        
       
        x=r.recognize_google(audio_text)
        print(x)
        
        return x
voiceconversion()