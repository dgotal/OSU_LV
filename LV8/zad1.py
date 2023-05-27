from tensorflow import keras
from keras import layers         #bug, radi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#ZADATAK 1 - Koliko primjera sadrži skup za učenje, a koliko skup za testiranje
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f'Broj primjera za ucenje: {len(X_train)}')
print(f'Broj primjera za testiranje: {len(X_test)}')
#moze se i skalirat sa X_train_s = X_train.astype("float32")/255
#ulazni podaci imaju oblik (broj primjera,28,28)(svaka slika je 28x28 piksela), svaki piksel predstavljen brojem 0-255
#izlazna velicina kodirana na nacin da su znamenke predstavljene brojevima 0-9
#svaka slika(primjer)-2d matrica, 28x28

#ZADATAK2
#prikaz random slike i oznake
X_train_reshaped = np.reshape(X_train,(len(X_train),X_train.shape[1]*X_train.shape[2])) #umjesto len(X_train) moze i X_train.shape[0]
X_test_reshaped = np.reshape(X_test,(len(X_test),X_test.shape[1]*X_test.shape[2]))      #umjesto len(X_test) moze i X_test.shape[0]
plt.imshow(X_train[7])   #slike se prikazuju normalnim 2d poljem
plt.title(f'Slika broja {y_train[7]}')
plt.show()

print("Oznaka slike:", y_train[7])
#ZADATAK3
#izrada mreze i ispis detalja
model = keras.Sequential()
model.add(layers.Input(shape=(784,))) #ulaz 784 elementa
model.add(layers.Dense(units=100, activation="relu")) #1. skriveni sloj 100 neurona "relu"
model.add(layers.Dense(units=50, activation="relu")) #2. skriveni sloj, 50 neurona, "relu"
model.add(layers.Dense(units=10, activation="softmax")) #Izlazni sloj 10 neurona, softmax
#pomoću metode summary ispišite informacije o mreži u terminal
model.summary()

#oneHotEncoding izlaza, da sve bude prema skici u predlosku, za ovo u kerasu postoji isto funkcija y_train = keras.utils.to_categorical(y_train, num_classes=10)
oh=OneHotEncoder()
y_train_encoded = oh.fit_transform(np.reshape(y_train,(-1,1))).toarray() #OneHotEncoder trazi 2d array, pa treba reshape (-1,1), tj (n,1),
y_test_encoded = oh.transform(np.reshape(y_test,(-1,1))).toarray() #-1 znaci sam skontaj koliko, mora toarray() obavezno kod onehotencodera

#ZADATAK4 Pomoću metode .compile podesite proces treniranja mreže
#podesavanje parametara treninga
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])

#ZADATAK5 Pokrenite učenje mreže (samostalno definirajte broj epoha i veličinu serije) 
history = model.fit(X_train_reshaped , y_train_encoded, batch_size=32, epochs=20, validation_split=0.1)

#ZADTAK6 Izvršite evaluaciju mreže na testnom skupu podataka pomoću metode .evaluate
#evaluacija i ispis 
score = model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

#ZADATAK7 Izračunajte predikciju mreže za skup podataka za testiranje + matrica zabune
#predict i matrica zabune
y_predictions = model.predict(X_test_reshaped)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (10 000,10)
y_predictions = np.argmax(y_predictions, axis=1)  #vraća polje indeksa najvecih elemenata u svakom pojedinom retku (1d polju) (0-9) (10 000,) - 1d polje
cm = confusion_matrix(y_test, y_predictions)    #zbog prethodnog koraka, usporedba s y_test, a ne encoded
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#ZADTAK 8 - spremite model
#spremanje modela
model.save('osu_lv/LV8/Model1/')
