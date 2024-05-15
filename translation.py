from googletrans import Translator
from gtts import gTTS
from pygame import mixer

class Translation:
	def __init__(self):
		self.translator = Translator()
		mixer.init()

	def get_translation(self, text, dest="zh-CN"):
		return self.translator.translate(text, dest=dest).text

	def play_tts(self, text="Something went wrong", lang="en"):
		mp3 = gTTS(text=text, lang=lang, slow=False)
		mp3.save("temp.mp3")
		mixer.music.load("temp.mp3")
		mixer.music.play()