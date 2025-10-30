# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: lite-venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deep-Learning: Ziffernerkennung mit KNN (Tiefe Netze)
# Nachdem wir im letzten Kapitel lediglich einziges Hidden Layer verwendet haben, erweitern wir dies nun auf sogenannte tiefe Netze, welche beliebig viele Hidden Layer enthalten können. Dieses Notebook ist sehr ähnlich zu dem aus dem letzten Kapitel, aber du kannst hier nun solche **tiefe Netze mit beliebig vielen Hidden Layern** erzeugen.

# %%
# Die notwendigen Bibliotheken werden geladen
import numpy as np
from scipy.special import expit as sigmoid
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# %% [markdown]
# ## Kantengewichte zufällig erzeugen

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 1
#
# Finde heraus, was der untenstehende Befehl tut. Du kannst dafür mit den 3 Eingabewerten etwas herum experimentieren. Halte deine Beobachtungen schriftlich fest.

# %%
np.random.normal(0.0, 0.1, (2, 3))


# %% [markdown]
# Beobachtungen bitte hierher

# %% [markdown]
# ## KNN als Klasse definieren
# Zunächst definieren wir eine Python-Klasse für unser künstliches neuronales Netz. Die Funktionen `__init__` und `knn_output` kannst du sicherlich schon im Detail verstehen.
#
# Die Funktion `train` wird zum Trainieren des KNN mittels Gradientenabstiegsverfahren verwendet. Die genaue Formulierung des Gradientenabstiegsverfahrens mit einer beliebigen Anzahl an Hidden-Layer-Neuronen ist etwas technischer und wird erst im nächsten Kapitel ausführlicher behandelt. Im Prinzip ist es aber genau dasselbe wie der weiter oben betrachtete Gradientenabstieg beim Beispiel mit den ungefährlichen und gefährlichen Tieren, nur halt mit sehr vielen Neuronen gleichzeitig.

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 2
#
# Beschreibe, was die beiden Funktionen `__init__` und `knn_output` tun. Liste auf, welche Parameter sie erhalten und was sie zurückgeben.

# %% [markdown]
# Ergebnisse bitte hierher

# %%
# Klassendefinition des Neuronalen Netzes
class neuralNetwork:
    # Initialisierung des Neuronalen Netzes
    def __init__(self, sizes, learningrate):
 
        # Gewichtsmatrizen udn Biasvektoren initialisieren
        self.weights = [np.random.normal(0.0, pow(x, -0.5), (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases  = [np.random.normal(0.0, pow(y,-0.5 ), (y, 1)) for y in sizes[1:]]
 
        # Lernrate
        self.lr = learningrate
 
        # Definition der Aktivierungsfunktion (hier als Sigmoid-Funktion)
        self.activation_function = lambda x: sigmoid(x)
 
        pass

    # Output des Künstlichen Neuronalen Netzes
    # NB: alle arrays sind stehende Vektoren!
    def knn_output(self, inputs_list):
        
        # Inputs in Array konvertieren 
        activation = np.array(inputs_list, ndmin=2).T # stehender Vektor

        # Berechnung der Aktivierung der Neuronen
        for w,b in zip(self.weights, self.biases):
            activation = np.dot(w, activation) + b
            activation = self.activation_function(activation)

        return activation
    

    # Gewichte des Neuronalen Netzes aktualisieren
    # siehe auch 
    # https://www.geeksforgeeks.org/backpropagation-in-machine-learning/
    def train(self, inputs_list, targets_list):

        # Inputs in 2-dimensionales Array konvertieren (Inputwerte und Zielwerte)
        inputs  = np.array(inputs_list, ndmin=2).T   # stehender Vektor
        targets = np.array(targets_list, ndmin=2).T # stehender Vektor
        
        activation = inputs
        activations = [inputs]
        zs=[]

        # Feedforward, abspeichern aller zs und activations, layer by layer
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        # Berechnung Fehler und Delta, zunächst Outputlayer
        errors = []
        deltas = []
        errors.insert(0, activations[-1]-targets )
        deltas.insert(0, errors[0] * activations[-1] * (1.0 - activations[-1]))

        # Berechnung der Fehler in den versteckten Schichten (Backpropagation)
        for i in range( 1, len(self.weights) ):
            #errors.insert(0, np.dot(self.weights[-i].T, errors[0]) ) #Version Yannik(?!)
            errors.insert(0, np.dot(self.weights[-i].T, deltas[0]) ) # Version Ulla
            deltas.insert(0, errors[0] * activations[-i-1] * (1.0 - activations[-i-1]))

        # Update der Kantengewichte mit Hilfe der Ableitung 
        # NB: activations[-3].T ist liegender Vektor!
        # activations[-2].T ist liegender Vektor!
        # NB: Hadamard-Produkte von stehenden Vektoren, Ergebnis ist ein stehender Vektor:
        #     (errors[-1] * activations[-1] * (1.0 - activations[-1])
        # NB: stehender mal liegender Vektor, Ergebnis ist Matrix:
        #    np.dot((errors[-1] * activations[-1] * (1.0 - activations[-1])), activations[-2].T)
        for i in range(1, len(self.weights)+1):
            self.weights[-i] -= self.lr * np.dot( deltas[-i], activations[-i-1].T )
            self.biases[-i]  -= self.lr * deltas[-i]

        pass

    
    

# %% [markdown]
# ## Festlegen der Parameter und Erstellen des KNN

# %%
# Anzahl der Neuronen in den einzelnen Schichten
# 28x28 = 784 Pixel-Bild als Input
#         100 künstl. Neuronen im ersten Hidden Layer
#          50 künstl. Neuronen im zweiten Hidden Layer
#          10 Ziffern (0-9) als Output
sizes = [784,100,50,10]

#Lernrate
learning_rate = 0.1

# Neuronales Netz erstellen
knn = neuralNetwork(sizes, learning_rate)

# %% [markdown]
# ## Trainingsdaten einlesen

# %%
# Einlesen der Trainingsdaten
training_data_file = open("mnist_dataset/mnist_train_1000.csv", 'r')
training_data_lines = training_data_file.readlines()
training_data_file.close()


# %% [markdown]
# ## Funktionen zum Plotten der Bilder

# %%
# data_line ist eine Zeile aus dem Datensatz,
# also ein String der Länge 1+28x28 = 1+784 = 785
def plot(data_line):
    pic_values = data_line.split(',')
    image= np.asarray(pic_values[1:], dtype=float).reshape((28,28))
    fig, axes = plt.subplots(figsize=(1,1))
    axes.matshow(image, cmap=plt.cm.binary)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()


# %%
# data_lines ist eine Liste von Zeilen aus dem Datensatz,
# also eine Liste von Strings der Länge 1+28x28 = 1+784 = 785
def plot_list(data_lines):
    images=[]
    for i in range(len(data_lines)):
        pics_values = data_lines[i].split(',')
        images.append(np.asarray(pics_values[1:], dtype=float).reshape((28,28)))
    fig, axes = plt.subplots(nrows=1, ncols=len(images))
    for j, ax in enumerate(axes):
        ax.matshow(images[j].reshape(28,28), cmap = plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# %%
plot(training_data_lines[0])

# %%
plot_list(training_data_lines[0:10])


# %% [markdown]
# ## Funktionen zur Vorhersage der Ziffern

# %%
# Vorhersage für einzelnes Bild
def predict( pic):
    #Inputdaten vorbereiten
    pic_values = pic.split(',')
    inputs = (np.asarray(pic_values[1:], dtype=float) / 255.0 * 0.99) 

    # Auswertung durch Neuronales Netz
    outputs = knn.knn_output(inputs)

    # Index mit dem höchsten Gewicht in der Ausgangsschicht sagt Ziffer vorher
    return np.argmax(outputs)


# %%
# Vorhersage für Lite von Bildern
def predict_list( pics ):
    predictions = []
    for pic in pics:
        #Inputdat   en vorbereiten
        pic_values = pic.split(',')
        # Inputdaten standardisieren
        inputs = (np.asarray(pic_values[1:], dtype=float) / 255.0 * 0.99) + 0.01

        # Auswertung durch Neuronales Netz
        outputs = knn.knn_output(inputs)

        # Index mit dem höchsten Gewicht in der Ausgangsschicht sagt Ziffer vorher
        predictions.append(int(np.argmax(outputs)))
    return predictions


# %%
plot(training_data_lines[0])
print(predict(training_data_lines[0]))

# %%
plot_list(training_data_lines[0:10])
print(predict_list(training_data_lines[0:10]))


# %% [markdown]
# ## Neuronales Netz vor dem Training Testen

# %%
# Liste falsch klassifizierte Bilder erzeugen
def wrong_list(data_list):
    wrong_list = []
    for i in range(len(data_list)):
        prediction = predict(data_list[i])
        if prediction != int(data_list[i].split(',')[0]):
            wrong_list.append(data_list[i])
    return wrong_list


# %%
wl = wrong_list(training_data_lines)
len(wrong_list(training_data_lines))

# %%
plot_list(wl[0:10])
print( predict_list(wl[0:10]) )
print( [int(wl[i].split(',')[0]) for i in range(len(wl[0:10]))] )

# %%
# Liste von falsch klassifizierte Bilder erzeugen
wl = wrong_list(training_data_lines)
print("Falsch klassifiziert:", len(wl))
print("Korrekt klassifiziert:", len(training_data_lines)-len(wl))
print("Anteil korrekt klassifiziert:", round((len(training_data_lines)-len(wl))/len(training_data_lines)*100,2), "%")

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 3
# Halte fest, was du beim Testen des KNN feststellst. Erkläre, warum das KNN die beobachtete Erkenntungsrate zeigt.
#

# %% [markdown]
# Ergebnisse bitte hier notieren.

# %% [markdown]
# ## Das KNN trainieren

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 4
# Stelle Vermutungen an, warum es in dem folgenden Trainingsprozess sinnvoll ist, die Input-Daten vor der Verarbeitung zu standardisieren.

# %% [markdown]
# Vermutungen bitte hierher

# %%
# Anzahl der Epochen
epochs = 5

# Schleife über die Epochenanzahl
for e in range(epochs):
    
    # nacheinander alle Einträge des Trainingsdatensatzes durchgehen
    for x in training_data_lines:
        #Einträge anhand des Komma splitten
        pic_values = x.split(',')
        
        #Inputs standardisieren
        inputs = (np.asarray(pic_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        
        # Ergebnisvektor erstellen
        targets = np.zeros(sizes[-1]) + 0.01
        # Nur der richtige Wert wird auf 0.99 gesetzt
        targets[int(pic_values[0])] = 0.99
        
        knn.train(inputs, targets)

# %%
# Liste von falsch klassifizierte Bilder erzeugen
wl = wrong_list(training_data_lines)
print("Falsch klassifiziert:", len(wl))
print("Korrekt klassifiziert:", len(training_data_lines)-len(wl))
print("Anteil korrekt klassifiziert:", round((len(training_data_lines)-len(wl))/len(training_data_lines)*100,2), "%")

# %%
# False klassifizierte Bilder anzeigen
plot_list(wl)
print( predict_list(wl) )
print( [int(wl[i].split(',')[0]) for i in range(len(wl))] )

# %%
# Einzelnes falsch klassifiziertes Bild anzeigen
number=0
data_lines = training_data_lines
plot(wrong_list(data_lines)[number])
print("Prediction:", predict(wrong_list(data_lines)[number]))
print("Classification", wrong_list(data_lines)[number].split(',')[0])

# %%
# Vorhersage für Lite von Bildern anzeigen
plot_list(training_data_lines[0:10])
predict_list(training_data_lines[0:10])

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 5
# Führe mehrere Trainingsepochen durch und beobachte, wie sich der Anteil an korrekt klassifizierten Bilder ändert. Halte deine Beobachtungen schriftlich fest.

# %% [markdown]
# Ergebnisse bitte hierher

# %% [markdown]
# ## Das KNN mit den Testdaten testen
# Nun wollen wir schauen, ob das fertig trainierte KNN auch mit neuen Bildern zurecht kommt, die es im Laufe des Trainings noch niemals "gesehen" hat.
#
# Dazu lesen wir zunächst die neuen Test-Daten aus einer Datei ein.

# %%
# Einlesen der Testdaten
test_data_file = open("mnist_dataset/mnist_test_100.csv", 'r')
test_data_lines = test_data_file.readlines()
test_data_file.close()

# %%
# Liste von korrekt klassifizierte Bilder erzeugen
correct_list = []
for i in range(len(test_data_lines)):
    prediction = predict(test_data_lines[i])
    if prediction == int(test_data_lines[i].split(',')[0]):
        correct_list.append(test_data_lines[i])

# %%
print( "Anzahl korrekt klassifiziert:", len(correct_list) )
print( "Anteil korrekt klassifiziert:", round(len(correct_list)/len(test_data_lines)*100,2) , "%" )

# %%
# Liste falsch klassifizierte Bilder erzeugen
wrong_list = []
for i in range(len(test_data_lines)):
    prediction = predict(test_data_lines[i])
    if prediction != int(test_data_lines[i].split(',')[0]):
        wrong_list.append(test_data_lines[i])

# %%
print("Anzahl falsch klassifiziert:", len(wrong_list))
plot_list(wrong_list)
print(predict_list(wrong_list))

# %%
# predictions anzeigen
for i in range(1):
    plot_list(test_data_lines[i*10:(i+1)*10])
    print( predict_list(test_data_lines[i*10:(i+1)*10]))

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 6
# Teste, wie gut das KNN auf den unbekannten Testdaten abschneidet. Stelle Vermutungen über die Grenzen unseres bisherigen Modells an.

# %% [markdown]
# Ergebnisse bitte hierher

# %% [markdown]
#
