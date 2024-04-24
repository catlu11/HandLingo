from tkinter import *
import cv2
from PIL import Image, ImageTk
from datetime import datetime

app = Tk()
app.title('HandLingo')
app.bind('<Escape>', lambda x: app.quit())

# Shows video feed and records when button is pressed 
recording = False
frames_list = []

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

def start_rec():
    global recordingx
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
languages_list = [
	'Afrikaans',
	'Albanian',
	'Chinese (simplified)',
	'French',
	'German',
	'Korean',
	'Latin',
	'Norwegian',
	'Serbian'
]
language_selected = StringVar(app)
language_selected.set('Unselected')
languages_dropdown = OptionMenu(app, language_selected, *languages_list)
languages_dropdown.pack()

# Translates gloss to desire languauge
def translate():
	print('translate to this language:')
	print(language_selected.get())

# Creates translate button
translate_button = Button(app, text='Translate', command=translate)
translate_button.pack()

# Creates gloss and translation log
log = Message(app, bd=2, bg='gray')
log.pack(expand=True, fill="both")

update_frame()

# Create an infinite loop for displaying app on screen
app.mainloop()