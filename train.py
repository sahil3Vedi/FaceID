# importing modules
import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')

import face_recognition
import cv2
import numpy as np
import os

# Folder in which image is saved is considered to be label

DATA_DIR = 'data'
TRAINING_DIR = 'training'
TESTING_DIR = 'testing'
TRAINING_PATH = os.path.join(DATA_DIR, TRAINING_DIR)
TESTING_PATH = os.path.join(DATA_DIR, TESTING_DIR)

names=[]
encodings=[]


# TRAINING NAMES AND ENCODING ASSOCIATION USING TRANSFER LEARNING
print("Beginning Training")
for filename in os.listdir(TRAINING_PATH):
  label = filename[:-4]
  FILE_PATH = os.path.join(TRAINING_PATH, filename)
  image_file = face_recognition.load_image_file(FILE_PATH)
  face_encoding = face_recognition.face_encodings(image_file)[0]
  names.append(label)
  encodings.append(face_encoding)

print("Training Complete. Starting Testing.")

correct_answers=0
total_answers=0
model_answer = []

# TESTING RESULTS
for each_folder in os.listdir(TESTING_PATH):
  target = each_folder
  TARGET_DIR = os.path.join(TESTING_PATH,each_folder)
  for each_file in os.listdir(TARGET_DIR):
    INPUT_PATH = os.path.join(TARGET_DIR,each_file)
    input_image = face_recognition.load_image_file(INPUT_PATH)
    output_encoding = face_recognition.face_encodings(input_image)[0]
    #calculating nearest distance among all known encodings
    face_distances = face_recognition.face_distance(encodings, output_encoding)
    matches = face_recognition.compare_faces(encodings, output_encoding)
    best_match_index = np.argmin(face_distances)
    answer = "Unknown"
    if matches[best_match_index]:
      answer = names[best_match_index]
    total_answers += 1
    if answer == target:
      correct_answers +=1
    print("Answer: " + answer + " "*(30-len(answer)) + "Target: " + target)

print("Accuracy: " + str(correct_answers) + "/" + str(total_answers))

