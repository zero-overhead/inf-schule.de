# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Der MNIST-Datensatz

# %% [markdown]
# Der **MNIST** (Modified National Institute of Standards) Datensatz ist ein Datensatz mit 70.000 Bildern von handgeschriebenen Ziffern. Dieser Datensatz wird weltweit als Standarddatensatz genutzt, um zu prüfen wie gut Machine Learning Verfahren die Bilderkennung beherrschen. Darüber hinaus zeigt es ein Anwendungbeispiel bei dem man mit herkömmlichen Programmiermethoden schnell an Grenzen stößt, das aber mit Künstlichen Neuronalen Netzen sehr gut lösbar ist.

# %% [markdown]
# <td> <img src="MNIST_Ziffern.png" alt="Drawing" style="width: 400px; float: left;"/> </td>
#
#

# %% [markdown]
# Bevor wir selbst ein KNN aufsetzen können, das anhand der MNIST Daten lernen und hinterher die Ziffern auf den Bildern erkennen kann, müssen wir erst einmal schauen, wie der MNIST Datensatz überhaupt aussieht. 
#
# In diesem Notebook werden daher einige Hilfsfunktionen zum Einlesen und zum Plotten der Bilder zur Verfügung gestellt. In den folgenden Kapiteln werden wir dann immer wieder auf diese Hilfsfunktionen zugrückgreifen.

# %% [markdown]
# ## Daten einlesen
# Zunächste lesen wir einmal einen kleinen Teil-Datensatz mit nur 100 Bildern von der Festplatte ein.

# %%
#Import der Bibliotheken
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# %%
# CSV-Datei öffnen und Inhalte in eine Liste laden
data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
data_lines = data_file.readlines()
data_file.close()

# %%
# Anzahl der Einträge checken
len(data_lines)

# %% [markdown]
# Jeder der 100 Einträge in der Datenliste stellt ein Bild einer handschriftlichen Ziffer dar, wie oben gezeigt.

# %%
# Ein Bild näher betrachten
pic_nr = 7

# %%
# Eintrag des Datensatzes ansehen
print( data_lines[pic_nr] )
print ( type(data_lines[pic_nr]) )


# %% [markdown]
# Der Eintrag aus <code>data_list</code> ist ein langer String der ganz viele **Zahlen zwischen 0 und 255** enthält. Um genau zu sein sind es **785 Zahlen**. Damit man sieht, dass es sich hier um ein Bild handelt muss man noch ein paar Schritte erledigen.

# %% [markdown]
# ## Daten analysieren
# Erstmal machen wir aus dem langen String eine Liste mit 785 Einträgen:

# %%
#String in Liste umwandeln
pic_values = data_lines[pic_nr].split(',')
print( pic_values )

# %%
print( len(pic_values ) )
print( 28*28 )

# %% [markdown]
# Der erste Eintrag von den 785 ist das Label des Bildes, d. h. er sagt welche Zahl dort abgebildet ist. Unser Bild soll also eine handschriftliche **3** zeigen. 
#
# Die weiteren 784 Werte stellen das Bild dar. Um zu sehen, wie das funktioniert, müssen wir die Daten aber noch etwas weiter verarbeiten. Es gilt $28\cdot 28 = 784$, d. h. wir können die Zahlen in einer **28x28 - Matrix** darstellen.
#
# Eine Matrix kann man in Python am besten mit der <code>numpy</code> Bibliothek erzeugen. Den ersten Zahlenwert wollen wir nicht mit einbeziehen und betrachten deshalb <code>pic_values[1:]</code> und stellen die restlichen 784 Zahlenwerten als **28x28**-Matrix dar. 

# %%
# Daten des ersten Datensatzes nehmen, als 28x28-Array umformtieren und plotten
# asarray: Konvertiert den Input zu einem Array, wobei der Typ float verwendet wird,
# dieser Typ wird im Folgenden benötigt, um die Werte mit matplotlib.imshow zu plotten
pic_array = np.asarray(pic_values[1:],dtype=np.float32).reshape((28,28)) 

# ASCII-Ausgabe der Bilddaten
for z in range(28):
    for s in range(28):
        # Ausgabe der Werte mit 3 Stellen, rechtsbündig
        print (str(int(pic_array[z][s])).rjust(3), end=' ')
    print()


# %% [markdown]
# Im der ASCII-Ausgabe des Array ist schon zu erkennen, dass es sich tatsächlich um ein Bild handelt.

# %% [markdown]
# ## Daten als Bild darstellen
# Jetzt wollen wir die Daten aber auch noch als Bild darstellen. Dafür gibt es den <code>imshow</code> Befehl aus der Bibliothek <code>matplotlib</code>. Durch den Parameter `cmap = 'Greys'` werden die Zahlen aus dem Array als Grauwerte interpretiert, wobei der Wert 0 ein weißes Pixel erzeugt, der Wert 255 ein schwarzes Pixel und die Werte dazwischen verschieden Abstufungen von grauen Pixeln.

# %%
imshow(pic_array, cmap='Greys')

# %% [markdown]
# Hier noch einmal zusammenfassend alle Schritte, um einen Datensatz als Bild dastzustellen:

# %%
#Schaue dir verschiedene Zahlen aus dem Datensatz an 
pic_nr=7
pic_values = data_lines[pic_nr].split(',')
pic_array = np.asarray(pic_values[1:],dtype=np.float32).reshape((28,28))
plt.figure(figsize=(1,1))
plt.xticks([])
plt.yticks([])
plt.imshow(pic_array, cmap='Greys')


# %% [markdown]
# ## Funktionen zum Plotten der Bilder
# Abschließend definieren wir noch zwei Funktionen, um die einglesenen String-Daten direkt plotten zu können.

# %%
# data_line ist eine Zeile aus dem Datensatz,
# also ein String der Länge 1+28x28 = 1+784
def plot_pic(data_line):
    pic_values = data_line.split(',')
    image = np.asarray(pic_values[1:],dtype=np.float32).reshape((28,28))
    fig, axes = plt.subplots(figsize=(1,1))
    axes.matshow(image, cmap=plt.cm.binary)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()


# %%
plot_pic(data_lines[7])


# %%
# data_lines ist eine Liste von Zeilen aus dem Datensatz,
# also eine Liste von Strings der Länge 1+28x28 = 1+784
def plot_pics(data_lines):
    images=[]
    for i in range(len(data_lines)):
        pics_values = data_lines[i].split(',')
        images.append(np.asarray(pics_values[1:],dtype=np.float32).reshape((28,28)))
    fig, axes = plt.subplots(nrows=1, ncols=len(images))
    for j, ax in enumerate(axes):
        ax.matshow(images[j].reshape(28,28), cmap = plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# %%
plot_pics(data_lines[0:10])
print( [int(data_lines[i][0]) for i in range(10)] )

# %%
for i in range(2):
    plot_pics(data_lines[i*10:(i+1)*10])
    print( [int(data_lines[i*10+j][0]) for j in range(10)] )

# %% [markdown]
# # Erzeugen einer Matrix von Bildern
# Manchmal ist es nützlich, die Bilder platzsparend in einer Matrix darzustellen. Deshalb schauen wir uns auch noch eine Möglichkeite an, wie man dies tun kann.

# %%
data_file = open("mnist_dataset/mnist_train_1000.csv", 'r')
data_lines = data_file.readlines()
data_file.close()


# %%
# Erzeugt eine 10x10-Matrix der ersten 100 Bilder des Triainingssatzes:
def plot_100_pics_as_matrix(data_lines):
    images=[]
    for i in range(len(data_lines)):
        pics_values = data_lines[i].split(',')
        images.append(np.asarray(pics_values[1:],dtype=np.float32).reshape((28,28)))
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(5, 5))
    for i in range(10):
        for j in range(10):
            axes[i,j].matshow(images[i*10+j], cmap = plt.cm.binary)
            axes[i, j].axis('off')
    plt.show()


# %%
plot_100_pics_as_matrix(data_lines[0:100])

# %%
