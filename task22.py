import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive
%matplotlib inline

drive.mount('/content/drive', force_remount=True) #подключаем гуглдиск

all_photo = np.empty(shape=(400, 10304), dtype=float)
numpos = []

count = 0

for fold in range(1,41):
  #count = 0
  for numPhoto in range(1,11):
    img = cv2.imread('/content/drive/My Drive/Colab/photo/s{folder}/{id}.pgm'.format(folder=fold,id=numPhoto)) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_photo[count][:] = gray.flatten() / 255
    numpos.append(count)
    count += 1

for fold in range(2,9):
  #human = all_photo[fold*10-10:fold*10][:]
  x_train, x_test, y_train, y_test = train_test_split(all_photo, numpos, test_size=fold/10, random_state=17)
  regr = linear_model.LinearRegression()
  regr.fit(x_train, y_train)
  # рассчитываем на основе тестовых данных
  pred = regr.predict(x_test)

  print('коэфициенты: \n', regr.coef_)
  print('среднеквадратичная ошибка: %.2f' % mean_squared_error(y_test, pred))
  print('Коэффициент точности определения: %.2f' % r2_score(y_test, pred))

#print(x_train)
#print(x_test)
##print(y_train)
#print(y_test)


#plt.imshow(img)