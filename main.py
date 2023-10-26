from roboflow import Roboflow
rf = Roboflow(api_key="vQ2fu2LUoE90OQ9apBw0")
project = rf.workspace().project("acnepredictions")
model = project.version(5).model

img = "cropped-shot-young-woman-s-face-before-after-acne-treatment-face-z_407348-18.webp"
conf = 20

# infer on a local image
#print(model.predict(img, confidence=conf, overlap=30).json())

# visualize your prediction
model.predict(img, confidence=conf, overlap=30).save("prediction.jpg")