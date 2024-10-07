import json
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo
from clarifai.rest import Image as ClImage

'''
try:
    import numpy as np
except ImportError:
    print( "numpy is not installed")
import cv2
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')


#get video from laptop web camera------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

self._name = name + '.mp4'
self._cap = VideoCapture(0)
self._fourcc = VideoWriter_fourcc(*'MP4V')
self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))

# video = ClVideo(filename='/home/user/video.mp4')
#image = ClImage(url='https://samples.clarifai.com/metro-north.jpg')
#END NEW CODE--------------------------------------------------------------------------------------------------
'''

import time
from SimpleCV import Camera
cam = Camera()
time.sleep(0.1)  # If you don't wait, the image will be dark
img = cam.getImage()
img.save("image.jpg")

occupied = False

app = ClarifaiApp(api_key='b31218b84a2142a99a6cd9776fa24def')
model = app.models.get('general-v1.3')
#image = ClImage(file_obj=open('image.jpg', 'rb'))
results = model.predict_by_filename('image.jpg', min_value=0.8, max_concepts=200)
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
results = model.predict_by_filename('imagep.jpg')
if results['outputs'][0]['data']: # face detected
	#print(results['outputs'][0]['data'])
	occupied = True

print(occupied)

# If not occupied, turn off light
