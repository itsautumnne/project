import json
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo
from clarifai.rest import Image as ClImage
import time
try:
    import numpy as np
except ImportError:
    print( "numpy is not installed")
import cv2
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

app = ClarifaiApp(api_key='b31218b84a2142a99a6cd9776fa24def')

model = app.models.get('general-v1.3')
# video = ClVideo(filename='/home/user/video.mp4')
#image = ClImage(url='https://samples.clarifai.com/metro-north.jpg')
#
# #get video from laptop web camera------------------------------------------------------------------------------------
# cap = cv2.VideoCapture(0)
#
# self._name = name + '.mp4'
# self._cap = VideoCapture(0)
# self._fourcc = VideoWriter_fourcc(*'MP4V')
# self._out = VideoWriter(self._name, self._fourcc, 20.0, (640,480))
# #END NEW CODE--------------------------------------------------------------------------------------------------
results = model.predict_by_filename('22.png', min_value=0.8, max_concepts=200)
fdict = {}
#create filtered dictionary of property names and their values
for i in range(len(results['outputs'][0]['data']['concepts'])):
    fdict[i] = results['outputs'][0]['data']['concepts'][i]

keys = []
values = []
for i in range(20):
    valueselem= str(fdict[i]['value'])
    keys.append(str(fdict[i]['name']))
    values.append(valueselem)

finaldict = dict(zip(keys, values))
print("Dictionary:" + str(finaldict))

#search for specific terms and then do things if they're found in the dictionary
#print("crowd" in [x for v in results['outputs'][0]['data']['concepts'][1] for x in v])
