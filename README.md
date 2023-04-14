# EmotionClassification_CNN
**Objective: Classification of emotion of a image.**

***Steps Involved:***
* Image `Data Collection` from keggle
* Training `Convolution Neural Network` and getting optimal model using `Tensorflow`
* Deployment of model with `FastAPI`
* Model version control with `Tensorflow serving`
* Building web app frontend with `ReackJS`

# How to use on your system?
***Requirements***
1. Intall all [requirements.txt](https://github.com/chanakyapk/EmotionClassification_CNN/blob/main/api/requirements.txt) file liberaries
2. Install [Docker](https://www.docker.com/)
3. Install [Tensorflow Serving](https://www.tensorflow.org/tfx/serving/setup), for model control
4. Install [NodeJS](https://nodejs.org/en/download), for ReactJS

***Steps***
- Download all files
- Run [main-tf-serving.py](https://github.com/chanakyapk/EmotionClassification_CNN/blob/main/api/main-tf-serving.py) with commands in api folder
- Run [docker commands](https://github.com/chanakyapk/EmotionClassification_CNN/blob/main/docker_commands.txt) of tensorflow-serving in command line
- Start ReactJS with `npm start run` in command line 

![image](https://user-images.githubusercontent.com/110924299/232023549-85eef541-d3c9-4145-991e-6dfed3ecd27f.png)

5. Drag and drop Image to Classify 

![image](https://user-images.githubusercontent.com/110924299/232023906-86844705-f638-4034-bce0-2830fcaae984.png)


