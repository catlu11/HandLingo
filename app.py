from tkinter import *
import cv2
from PIL import Image, ImageTk

###### COMMANDS ######

# Create a function to open camera and display it in the label_widget on app
def open_camera():

	# Capture the video frame by frame
	_, frame = vid.read()

	# Convert image from one color space to other
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

	# Capture the latest frame and transform to image
	captured_image = Image.fromarray(opencv_image)

	# Convert captured image to photoimage
	photo_image = ImageTk.PhotoImage(image=captured_image)

	# Displaying photoimage in the label
	video_label.photo_image = photo_image

	# Configure image in the label
	video_label.configure(image=photo_image)

	# Repeat the same process after every 10 seconds
	video_label.after(40, open_camera)

# Starts video recording
def start_recording():
	print('video started recording')

# Stops video recording
def stop_recording():
	print('video stopped recording')

# Translates gloss to desire languauge
def translate():
	print('translate gloss')


###### Application ######

app = Tk()
app.title('HandLingo')
app.geometry('800x600')
app.bind('<Escape>', lambda x: app.quit())

# Creates label for video capture
video_label = Label(app)
video_label.pack(side='left', padx=5, pady=5, expand=True)
vid = cv2.VideoCapture(0)
width, height = 350, 500 # Declare the width and height in variables
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) # Set the width and height
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

open_camera()

# Creates buttons for video recording
start_button = Button(app, text='Start Recording', command=start_recording)
start_button.pack()
stop_button = Button(app, text='Stop Recording', command=stop_recording)
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

# Creates translate button
translate_button = Button(app, text='Translate', command=translate)
translate_button.pack()

# Creates gloss and translation log
log = Message(app, bd=2, bg='gray')
log.pack()

# Create an infinite loop for displaying app on screen
app.mainloop()
