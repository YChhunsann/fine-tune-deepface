from deepface import DeepFace

result = DeepFace.analyze("crying1.jpg", actions=("emotion"))
print(result)