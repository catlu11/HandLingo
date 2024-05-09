from tkinter import *
import customtkinter
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from gloss_recognition.Transformer_nih import Sign2TextModel
from translation import Translation
import numpy as np

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk()
app.title('HandLingo')
# app.geometry('800x700')
app.bind('<Escape>', lambda x: app.quit())

app.grid_columnconfigure((4, 5), weight=1)

video_label = customtkinter.CTkLabel(app, text='')
video_label.grid(row=1, column=0, columnspan=4, padx=5, pady=5)
capture = cv2.VideoCapture(0)

recording = False
recordings_queue = []
frames_list = []
thumbnail_labels = []
tn_row = 2
tn_col = 0

thumbnail_frame = customtkinter.CTkFrame(app)
thumbnail_frame.grid(row=2, column=0, columnspan=4, rowspan=2, padx=5, pady=5, sticky="nsew")

log_text = 'Log: '
gloss_model = Sign2TextModel()
translator = Translation()

def update_frame():
	_, frame = capture.read()
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	captured_image = Image.fromarray(opencv_image)
	captured_image = captured_image.resize((600, 400))
	if recording:
		frames_list.append(frame)
	tk_img = ImageTk.PhotoImage(image=captured_image)
	video_label.tk_img = tk_img
	video_label.configure(image=tk_img)
	app.after(40, update_frame)

def start_rec():
    global recording, frames_list
    frames_list = []
    recording = True

    start_button.configure(state="disabled")
    stop_button.configure(state="normal")

def stop_rec():
	global recording, thumbnail_labels, tn_row, tn_col
	recording = False

	start_button.configure(state="normal")
	stop_button.configure(state="disabled")

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
	tn_label = customtkinter.CTkLabel(app, text='')
	tn_label.grid(row=tn_row, column=tn_col)
	if (tn_col + 1) < 4:
		tn_col += 1
	else:
		tn_row += 1
		tn_col = 0
	thumbnail_frame = cv2.resize(frames_list[len(frames_list)//2], (120, 80))
	opencv_thumbnail = cv2.cvtColor(thumbnail_frame, cv2.COLOR_BGR2RGBA)
	captured_thumbnail = Image.fromarray(opencv_thumbnail)
	tk_thumbnail = ImageTk.PhotoImage(image=captured_thumbnail)
	tn_label.tk_thumbnail = tk_thumbnail
	tn_label.configure(image=tk_thumbnail)
	thumbnail_labels.append(tn_label)

	print(len(thumbnail_labels))
	
def closeWindow():
    stop_rec()
    app.destroy()

start_button = customtkinter.CTkButton(app, width=280, text="Start Signing", command=start_rec)
start_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='ns')
stop_button = customtkinter.CTkButton(app, width=280, text='Stop Signing', command=stop_rec, state="disabled")
stop_button.grid(row=0, column=2, columnspan=2, padx=5, pady=5, sticky='ns')

# Translates gloss to desire languauge
def translate():
	global log_text, recordings_queue, thumbnail_labels, tn_row, tn_col
	gloss_txt = ''

	for thumbnail in thumbnail_labels:
		thumbnail.destroy()
	thumbnail_labels = []
	tn_row = 2
	tn_col = 0

	if len(recordings_queue) == 0:
		log_text += '\n\nNothing to translate'
		log.configure(text=log_text)

	for video_path in recordings_queue:
		gloss = gloss_model.get_prediction(video_path).strip()
		gloss_txt += gloss + ' '
	gloss_txt.strip()

	lang = language_selected.get()
	translated_txt = translator.get_translation(gloss_txt, dest=languages_list[lang]['trans'])

	log_text += '\n\nEnglish: ' + gloss_txt
	log_text += '\n' + language_selected.get() + ': ' + translated_txt
	log.configure(text=log_text)
	translator.play_tts(translated_txt, languages_list[lang]['tts'])

	recordings_queue.clear()

languages_list = {'Chinese (simplified)': {"trans": 'zh-CN', "tts": 'zh'}, 'Latin': {"trans": "la", "tts": "en"}}
language_selected = customtkinter.StringVar(value="Select Language")
languages_dropdown = customtkinter.CTkOptionMenu(app, width=280, values=list(languages_list.keys()), variable=language_selected)
languages_dropdown.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='ns')

translate_button = customtkinter.CTkButton(app, width=280, text='Translate', command=translate)
translate_button.grid(row=4, column=2, columnspan=2, padx=5, pady=5, sticky='ns')

# Creates gloss and translation log
log = customtkinter.CTkLabel(app, width=400, text=log_text, anchor='nw', justify="left")
log.grid(row=1, column=4, columnspan=5, rowspan=5, padx=5, pady=5, sticky='nsew')

# thumbnail placeholder
for i in range(2, 4):
	for j in range(4):
		tn_placeholder_label = customtkinter.CTkLabel(app, text='')
		tn_placeholder_label.grid(row=i, column=j)
		array = np.full((80, 120, 3), 43, dtype=np.uint8)
		img = Image.fromarray(array)
		tn_placeholder = ImageTk.PhotoImage(image=img)
		tn_placeholder_label.tn_placeholder = tn_placeholder
		tn_placeholder_label.configure(image=tn_placeholder, width=120, height=80)

update_frame()

app.mainloop()