import json
import dlib
import cv2
import matplotlib.pyplot as plt
from google.colab import drive
%matplotlib inline

drive.mount('/content/drive', force_remount=True) #подключаем гуглдиск

coordinates = {}

f = open("coordsfile.json", "a")

#1-40
for fold in range(1,41):

  for numPhoto in range(1,11):
    saveLeftEyePositionX = 0
    saveLeftEyePositionY = 0
    saveRightEyePositionX = 0
    saveRightEyePositionY = 0
    
    img = cv2.imread('/content/drive/My Drive/Colab/photo/s{folder}/{id}.pgm'.format(folder=fold,id=numPhoto)) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/content/drive/My Drive/Colab/files/shape_predictor_68_face_landmarks.dat')

    faces = detector(gray)
    for face in faces:
      landmarks = predictor(gray, face)

      #перебор всех маркеров 
      for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

        if n==45:
          print("{LeftEyeX}_{LeftEyeY}".format(LeftEyeX=x,LeftEyeY=y))
          saveLeftEyePositionX = x
          saveLeftEyePositionY = y
        elif n==36:
          print("{RightEyeX}_{RightEyeY}".format(RightEyeX=x,RightEyeY=y))
          saveRightEyePositionX = x
          saveRightEyePositionY = y
          
        coordinates.update({
          str(fold) + '/' + str(numPhoto): {
              "LeftEyeX": saveLeftEyePositionX,
              "LeftEyeY": saveLeftEyePositionY,
              "RightEyeX": saveRightEyePositionX,
              "RightEyeY": saveRightEyePositionY
          }
        })

          # the result is a JSON string:
coordinates = json.dumps(coordinates)

f.write(coordinates)
print(coordinates)

f.close()
plt.imshow(img)
