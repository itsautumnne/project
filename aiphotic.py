import json
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo
from clarifai.rest import Image as ClImage
import time
import numpy as np
# try:
#     import numpy as np
# except ImportError:
#     print( "numpy is not installed")
import cv2
import os


occupied = False

#__________________________________________________________________________________________________
camera = cv2.VideoCapture(0)

#function provided at https://stackoverflow.com/questions/30136257/how-to-get-image-from-video-using-opencv-python
def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    framerate = vidcap.get(10) #get framerate
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read(5)
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video_writer = cv2.VideoWriter('output.m4v', fourcc, 30.0, (640, 480))

#function provided at https://stackoverflow.com/questions/43448456/how-to-store-webcam-video-with-opencv-in-python
while True:
        (grabbed, frame) = camera.read()  # grab the current frame
        frame = cv2.resize(frame, (640,480))
        cv2.imshow("Frame", frame)  # show the frame to our screen
        key = cv2.waitKey(33) & 0xFF  # I don't really have an idea what this does, but it works..
        video_writer.write(frame)  # Write the video to the file system
        if key==27:
            break;

# cleanup the camera and close any open windows
camera.release()
video_writer.release()
cv2.destroyAllWindows()
print("\n\nBye bye\n")
video_to_frames('output.m4v', '/Users/xy/Documents/Hackathons/hackNY')

#____________________________________________________________________________________

app = ClarifaiApp(api_key='b31218b84a2142a99a6cd9776fa24def')
model = app.models.get('general-v1.3')
results = model.predict_by_filename('1.png', min_value=0.8, max_concepts=200)
tags = []

# get all concept names from results
for i in range(len(results['outputs'][0]['data']['concepts'])):
    tags.append(results['outputs'][0]['data']['concepts'][i]['name'])

#print tags
person = {'man', 'woman', 'child', 'girl', 'boy', 'portrait', 'model', 'adolescent', 'people', 'person',
	'meeting', 'adult', 'crowd', 'group together', 'audience', 'facial expression'}
# whether or not a person-related concept is in the tagged concpets
if not set(tags).isdisjoint(person):
	occupied = True


# Check with Face Detection model
model = app.models.get('face-v1.3')
results = model.predict_by_filename('1.png')
if results['outputs'][0]['data']: # face detected
	print(results['outputs'][0]['data'])
	occupied = True

print(occupied)
