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
# # Training von Entscheidungsbäumen
# Im letzten Kapitel wurden alle Daten für das Training des Entscheidungsbaums verwendet. Getestet wurde im Anschluss mit den selben Daten. Ob der Baum für andere Daten gute Ergebnisse liefert, wurde nicht überprüft.
# In diesem Kapitel sollen die eingelesenen Daten (möglichst zufällig) in **drei Gruppen** aufgeteilt werden:
# - **Trainingsdaten**: Mit diesen Datensätzen soll der Baum berechnet werden.
# - **Validierungsdaten**: Mit diesen Datensätzen soll geprüft werden, ob der Baum gut funktioniert. Wenn sich hierbei herausstellt, dass das Modell noch nicht gut genug ist, dann werden die so genannten *Hyperparamter* (s.u.) abgeändert und der Trainigsprozess wird erneut durchgeführt.
# - **Testdaten**: Mit diesen Daten wird der Anwendungsfall simuliert. Dieser findet erst **nach** der Trainingsphase statt, wenn die KI fertig ist und eingesetzt werden soll. 
#
# Anhand der Testdaten wird abschließend entschieden, ob das Modell für den vorgesehenen Einsatzzweck akzeptiert wird. Die Testdaten werden also niemals während des iterativen Trainings des Modells verwendet. Auf diese Weise soll ein so genanntes *Overfitting* des Modells an die Trainigs- und Validierungddaten erkannt und vermieden werden.
#
# Eine Aufteilung in diese 3 Gruppen macht nur Sinn, wenn viele Daten vorhanden sind. Deshalb werden hier immer alle 55 Lebensmittel (**d55.csv**) verwendet.

# %% [markdown]
# <center>
# <img width="60%" src="trainingsprozess.png"/>
# <center>

# %% [markdown]
# ## Einlesen der Daten und erster Überblick

# %%
import pandas as pd
daten = pd.read_csv('d55.csv')
from IPython.display import display
display(daten.head(5))   # Anzeige der ersten n Datensätze

# %% [markdown]
# ## Aufteilung der Daten in zufällige Gruppen
# Im folgenden Quelltext kann man die **Größe der einzelnen Gruppen angeben**. Die Summe muss dabei natürlich der Anzahl der Datensätze entsprechen. 
#
# Oft wird für die Aufteilungverhältnis der Trainigs-, Validierungs und Testdaten von 70% : 15% : 15% gewählt.

# %%
anzahl_training    = 35
anzahl_validierung = 15
anzahl_test        = len(daten) - anzahl_training - anzahl_validierung   # der Rest
print( "Aufteilungsverhältnis:", 
       round(anzahl_training/len(daten)*100,1) ,"% :",
       round(anzahl_validierung/len(daten)*100,1) ,"% :",
       round(anzahl_test/len(daten)*100,1) ,"%" )

# %% [markdown]
# Für die zufällige Verteilung auf die drei Gruppen erzeugen wir eine **Zufallsliste** mit den Nummern der Datensätze, in der die Elemente mit **random.shuffle** durchgemischt werden.
# Du kannst diesen Block mehrmals ausführen und dabei das wiederholte Neumischen in der Ausgabe beobachten.

# %%
import random
zufallsliste= list( range(len(daten)) ) # Liste mit den Nummern von 0 bis Anzahl der Datensätze
random.shuffle( zufallsliste )          # Liste wird zufällig durchgeschüttelt
print( zufallsliste )                   # Ausgabe zur Kontrolle

# %% [markdown]
# Jetzt müssen noch die **Daten in die einzelnen Gruppen** aufgeteilt werden. Dazu verwenden wir natürlich die Zufallsliste.

# %%
daten_training    = daten.loc[ zufallsliste[:anzahl_training] ] 
daten_validierung = daten.loc[ zufallsliste[anzahl_training:anzahl_training+anzahl_validierung] ] 
daten_test        = daten.loc[ zufallsliste[anzahl_training+anzahl_validierung:] ]
#print(daten_training['Name'])

# %% [markdown]
# ## Training des Entscheidungsbaums
# Wie im vorigen Kapitel können jetzt die **Parameter der Berechnung** und der Darstellung angepasst werden.

# %%
import sklearn.tree
import matplotlib

attribute = ['Eiweiss','Salz']
baumtiefe = 2

baum = sklearn.tree.DecisionTreeClassifier(max_depth=baumtiefe)
baum.fit( daten_training[attribute], daten_training['Label'] )
sklearn.tree.plot_tree( baum, feature_names=attribute, label="none", 
                        filled=True,
                        class_names=["ungesund","gesund"], impurity=False,
                        proportion=False, fontsize=8 )
matplotlib.pyplot.show()


# %% [markdown]
# ## Validierung
# Die Validierung wird nun mit den **Validierungsdaten** durchgeführt. 

# %%
def validierung(datenliste):
  korrekt = 0
  falsch = 0
  for datensatz in datenliste.index:  #Schleife über alle Validierungs-Datensätze
      if( baum.predict(datenliste.loc[[datensatz]][attribute]) == (datenliste.loc[datensatz]['Label']) ):  
           korrekt += 1
      else:
          falsch += 1
          print("Fehler bei", datenliste.loc[datensatz]['Name'])
  gesamt = korrekt + falsch
  print( "Anzahl falsch klassifiziert: ",  falsch, "(" , round(falsch/gesamt*100,2), "% )")  
  print( "Anzahl korrekt klassifiziert: ", korrekt, "(" , round(korrekt/gesamt*100,2), "% )") 



# %%
print("**************  V A L I D I E R U N G  *************")
validierung(daten_validierung)


# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 1: Iterative Verbesserung des Modells
# Gehe nun zurück zum Abschnitt [Training des Entscheidungsbaums](##Training-des-Entscheidungsbaums) und verändere die so genannten *Hyperparameter*
# - atrribute
# - baumtiefe
# - Aufteilungsverhältnis der Daten in Traings-, Validierung- und Testdaten
#
# solange, bis du zufrieden bist mit dem Ergebnis.

# %% [markdown]
# ---
# # Akzeptanz-Test
# Erst **nach** dem vollständigen Abschluss des iterativen Trainigsprozesses wird nur die Akzeptanzkontrolle mittels der **Testdaten** durchgeführt. Beachte, dass die Testdaten nun völlig neu für unser Modell sind, da wir diese im Trainingsprozess niemals verwendet haben.

# %%
print("**************      T E S T U N G      *************")
validierung(daten_test)

# %% [markdown]
# Wenn das Erbebnis zufriedenstellend ist, dann wird das nun fertig trainierte und getestete Modell im produktiven Einsatz verwedendet.
#
# Andernfalls wird es verworfen und es wird ein anderes Modell entworfen.

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 2: Beurteile dein fertiges Modell
# Beurteile dein fertiges Modell. Falls du nicht zufrieden damit bist, dann äußere Vermutungen darüber, was schief gegangen sein könnte.

# %% [markdown]
# Ergebnisse bitte hierher

# %% [markdown]
# ---
# # Aufbereitung der Daten

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 3: Aufbereitung der Daten
# Arbeite die folgende Seite über den Prozess der [Aufbereitung der Daten](https://mlu-explain.github.io/train-test-validation/) durch. Halte schriftlich fest, welche Aspekte dort zusätzlich erwähnt werden, die wir in diesem Notebook aber bisher überhaupt nicht berücksichtigt haben.
#

# %% [markdown]
# Ergebnisse bitte hier festhalten

# %% [markdown]
#
