# SWAG
Open Source Image Recognition

```python 
from imageai.Classification import ImageClassification

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("resnet50_imagenet_tf.2.0.h5")
prediction.loadModel()

predictions, probabilities = prediction.classifyImage("people/train/cops/IMG_3743.jpg", result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
    
```

We see no need to recreate the wheel. A Google Tensorflow workflow can look through images and derive the proper label. 

``` json

Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
40960/35363 [==================================] - 0s 1us/step
military_uniform  :  8.621860295534134
Newfoundland  :  5.858585610985756
mountain_bike  :  5.305056273937225
bulletproof_vest  :  4.944176599383354
dogsled  :  4.822985082864761


```
