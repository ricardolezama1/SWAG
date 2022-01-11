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
