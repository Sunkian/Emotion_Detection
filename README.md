# AI_features

AI-features is a program for facial recognition of one or more people.
It allows access to certain data such as geometric features of the face or the feeling experienced by the subject. It also displays on the webcam screen the date, a timer, and the dimension of the frame. Percentages of the subjet's 'emotion' are visible too.

## Getting started

The present repository is a Flask-RESTful API that follows the client-server architecture constraint. The main purpose was to detect, recognize and extract AI-based features and send them to a localhost through an API in order to collect data in a JSON format.

1. Download the zip
2. In the live.feature.py file, change the local paths on lines 115 and 118 to :

- model = load_model('/Users/'your_user_name/Download/AI-features/Models/video.h5')
- predictor_landmarks = dlib.shape_predictor(
  "/Users/'your_user_name'/Downloads/AI-features/Models/face_landmarks.dat")

3. On a terminal windown, start the server (app.py):

```
cd /Users/'your_user_name'/Downloads/AI-features
```

```
python3 app.py
```

4. On another terminal window, start the client (live_features.py)

```
cd /Users/'your_user_name'/Downloads/AI-features
```

```
python3 live_features.py
```

## Results

### Storing

All the data will be stored each frame in a JSON format in the 'Store' folder with the following names :

```python
filename = 'Store/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.json'
```

#### Understand Storing

Let's take an example.

```json
[
  {
    "timestamp": "2020-11-10 11:34:07.235271",
    "timer": "6.295633220672608",
    "features": {
      "geometry": {
        "jaw": [
          [515, 454],
          [519, 494],
          [525, 535],
          [535, 573],
          [551, 608],
          [575, 636],
          [604, 659],
          [636, 676],
          [671, 679],
          [704, 671],
          [730, 650],
          [755, 625],
          [774, 594],
          [785, 560],
          [790, 524],
          [793, 486],
          [794, 447]
        ],
        "right eyebrow": [
          [541, 431],
          [560, 417],
          [585, 413],
          [609, 419],
          [631, 429]
        ],
        "left eyebrow": [
          [686, 426],
          [709, 415],
          [733, 408],
          [758, 410],
          [775, 425]
        ]
      },
      "sentiments": {
        "angry": "0.07550531",
        "disgust": "0.00036768967",
        "fear": "0.08752836",
        "happy": "0.00587192",
        "sad": "0.26025492",
        "surprise": "0.009125197",
        "neutral": "0.56134653"
      }
    }
  }
]
```

The list of numbers in the "jaw" key represent the [x,y] coordinates of each landmarks in the jaw region. The same can be said for the other data related to the geometry category. According to the cv2 library, the jaw gathers 17 facial landmarks (points in red in the live stream result pictures), so there are 17 [x,y] positions in the JSON file.

```python
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
```

![Alt text](https://cto-github.cisco.com/futurecollab/AI_features/blob/master/Results/facial_landmarks_68markup.jpg "Facial Landmarks")

### Live stream results

![Alt text](https://cto-github.cisco.com/futurecollab/AI_features/blob/master/Results/Happy.png "Sentiment : Happy")
![Alt text](https://cto-github.cisco.com/futurecollab/AI_features/blob/master/Results/Surprised.png "Sentiment : Surprise")
![Alt text](https://cto-github.cisco.com/futurecollab/AI_features/blob/master/Results/Neutral.png "Sentiment : Neutral")

Ref : https://maelfabien.github.io/project/poleemploi/#