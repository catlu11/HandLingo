from tkinter import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from gloss_recognition.Transformer_nih import Sign2TextModel
from translation import Translation

app = Tk()
app.title('HandLingo')
app.bind('<Escape>', lambda x: app.quit())

video_label = Label(app)
video_label.pack(side='left', padx=5, pady=5)
capture = cv2.VideoCapture(0)

recording = False
recordings_queue = []
frames_list = []
thumbnail_labels = []

log_text = 'Log: '
gloss_model = Sign2TextModel()
translator = Translation()

def update_frame():
	_, frame = capture.read()
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	captured_image = Image.fromarray(opencv_image)
	if recording:
		frames_list.append(frame)
	tk_img = ImageTk.PhotoImage(image=captured_image)
	video_label.tk_img = tk_img
	video_label.config(image=tk_img)
	app.after(40, update_frame)

def stop_rec():
	global recording, thumbnail_labels
	recording = False

	start_button.config(state="normal")
	stop_button.config(state="disabled")

	now = datetime.now()
	dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
	video_path = 'videos/' + dt_string + '.mp4'
	recordings_queue.append(video_path)
	fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
	video_writer = cv2.VideoWriter(video_path, fourcc, 11.0, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
	for frame in frames_list:
		video_writer.write(frame)
	video_writer.release()

	#create thumbnail label
	tn_label = Label(app)
	tn_label.pack(side='left')
	thumbnail_frame = cv2.resize(frames_list[len(frames_list)//2], (60, 40)) 
	opencv_thumbnail = cv2.cvtColor(thumbnail_frame, cv2.COLOR_BGR2RGBA)
	captured_thumbnail = Image.fromarray(opencv_thumbnail)
	tk_thumbnail = ImageTk.PhotoImage(image=captured_thumbnail)
	tn_label.tk_thumbnail = tk_thumbnail
	tn_label.config(image=tk_thumbnail, width=60, height=40)
	thumbnail_labels.append(tn_label)

	print(len(thumbnail_labels))

def start_rec():
    global recording, frames_list
    frames_list = []
    recording = True

    start_button.config(state="disabled")
    stop_button.config(state="normal")

def closeWindow():
    stop_rec()
    app.destroy()

# Creates buttons for video recording
start_button = Button(app, text='Start Signing', command=start_rec)
start_button.pack()
stop_button = Button(app, text='Stop Signing', command=stop_rec, state="disabled")
stop_button.pack()

# Creates language selector
languages_list = {'Chinese (simplified)': {"trans": 'zh-CN', "tts": 'zh'}, 'Latin': {"trans": "la", "tts": "en"}}
language_selected = StringVar(app)
language_selected.set('Chinese (simplified)')
languages_dropdown = OptionMenu(app, language_selected, *languages_list.keys())
languages_dropdown.pack()

# Translates gloss to desire languauge
def translate():
	global log_text, recordings_queue, thumbnail_labels
	gloss_txt = ''

	for thumbnail in thumbnail_labels:
		thumbnail.destroy()
	thumbnail_labels = []

	# print(recordings_queue)
	if len(recordings_queue) == 0:
		log_text += '\n\nNothing to translate'
		log_text_var.set(log_text)

	for video_path in recordings_queue:
		gloss = gloss_model.get_prediction(video_path).strip()
		gloss_txt += gloss + ' '
	gloss_txt.strip()

	lang = language_selected.get()
	translated_txt = translator.get_translation(gloss_txt, dest=languages_list[lang]['trans'])

	log_text += '\n\nEnglish: ' + gloss_txt
	log_text += '\n' + language_selected.get() + ': ' + translated_txt
	log_text_var.set(log_text)
	translator.play_tts(translated_txt, languages_list[lang]['tts'])

	recordings_queue.clear()

# Creates translate button
translate_button = Button(app, text='Translate', command=translate)
translate_button.pack()

# Creates gloss and translation log
log_text_var = StringVar()
log = Label(app, textvariable=log_text_var, anchor='w', justify="left")
log_text_var.set(log_text)
log.pack(fill='both')

update_frame()

# Create an infinite loop for displaying app on screen
app.mainloop()
