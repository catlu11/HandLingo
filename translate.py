from googletrans import Translator

class Translation:
	def __init__(self):
		self.translator = Translator()

	def get_translation(self, text, dest="zh-CN"):
		return self.translator.translate(text, dest=dest).text

