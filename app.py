from tkinter import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from gloss_recognition.I3D_pretrained import Sign2TextModel
from translation import Translation

gloss_model = Sign2TextModel()
translator = Translation()
app = Tk()
app.title('HandLingo')
app.bind('<Escape>', lambda x: app.quit())

# Shows video feed and records when button is pressed 
recording = False
frames_list = []
last_gloss = None

video_label = Label(app)
video_label.pack(side='left', padx=5, pady=5)
capture = cv2.VideoCapture(0)

rand_counter = 0

def update_frame():
	global rand_counter
	_, frame = capture.read()
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	captured_image = Image.fromarray(opencv_image)
	if recording:
		rand_counter += 1
		frames_list.append(frame)
	tk_img = ImageTk.PhotoImage(image=captured_image)
	video_label.tk_img = tk_img
	video_label.config(image=tk_img)
	app.after(40, update_frame)

def stop_rec():
	global recording
	recording = False

	start_button.config(state="normal")
	stop_button.config(state="disabled")

	now = datetime.now()
	dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
	video_path = 'videos/' + dt_string + '.mp4'
	fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
	video_writer = cv2.VideoWriter(video_path, fourcc, 11.0, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
	for frame in frames_list:
		video_writer.write(frame)
	video_writer.release()

	global last_gloss
	last_gloss = gloss_model.get_prediction(video_path)
	print(last_gloss)

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
start_button = Button(app, text='Start Recording', command=start_rec)
start_button.pack()
stop_button = Button(app, text='Stop Recording', command=stop_rec, state="disabled")
stop_button.pack()

# Creates language selector
languages_list = {'Chinese (simplified)': 'zh-CN', 'Latin': 'la'}
language_selected = StringVar(app)
language_selected.set('Chinese (simplified)')
languages_dropdown = OptionMenu(app, language_selected, *languages_list.keys())
languages_dropdown.pack()

# Translates gloss to desire languauge
def translate():
	if last_gloss is None:
		print("Nothing to translate")
	else:
		lang = language_selected.get()
		translated_txt = translator.get_translation(last_gloss, dest=languages_list[lang])
		print(translated_txt)

# Creates translate button
translate_button = Button(app, text='Translate', command=translate)
translate_button.pack()

# Creates gloss and translation log
log = Message(app, bd=2, bg='gray')
log.pack(expand=True, fill="both")

update_frame()

# Create an infinite loop for displaying app on screen
app.mainloop()
