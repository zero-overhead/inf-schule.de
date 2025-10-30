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
# # Entscheidungsbäume mit Python berechnen lassen
# Dieses Jupyter-Notebook verwendet die gleichen Daten wie die Kids-Lernstrecke [Entscheide wie eine KI](https://www.inf-schule.de/kids/computerinalltag/entscheide-wie-eine-KI) auf inf-schule.de. Falls du diese Lernstrecke noch nicht kennst, dann arbeite sie vorher durch, um dich in die Thematik einzufinden. Anschließend kannst du dann mit diesem Notebook Entscheidungsbäume völlig automatisiert erstellen lassen.
#
# ## Was kann ich hier tun?
# Dieses Notebook verwendet verschiedene **Python-Bibliotheken** zur Erstellung und Anzeige von Entscheidungsbäumen:
# - **pandas**: Die Daten aus CSV-Dateien einlesen und sichten
# - **sklearn**: Entscheidungsbäume zu den Daten berechnen
# - **matplotlib**: Entscheidungsbäume grafisch darstellen
#
# Die folgenden Dateien stehen zum Einlesen zur Verfügung. Sie enthalten Daten von Lebensmitteln.
# - **d14.csv**:   Daten von 14 ausgewählten Lebensmitteln
# - **d28.csv**:   Daten von 28 ausgewählten Lebensmitteln
# - **d55.csv**:   Daten von allen 55 Lebensmitteln
#

# %% [markdown]
# # Kapitel 1: Daten aus der CSV-Datei einlesen

# %%
import pandas as pd
daten = pd.read_csv('d14.csv')  # Einlesen der Daten, hier kann man den Dateinamen anpassen.

# %%
from IPython.display import display
display(daten)

# %% [markdown]
# # Kapitel 2: Den Entscheidungsbaum berechnen lassen und darstellen

# %%
import sklearn.tree
import matplotlib

# %% [markdown]
# ## Parameter für die Entscheidungsfindung wählen
# Für die Berechnung des Entscheidungsbaums müssen einige Vorgaben gemacht werden. Im Einzelnen musst du dem System folgende Informationen mitteilen:
# - **Attribute**: Definiere, welche Eigenschaften der Lebensmittel zum Berechnen des Entscheidungsbaums verwendet werden sollen.
# - **Zielkriterium**: Definiere, welche Eigenschaft durch den Baum entschieden werden soll.
# - **Baumtiefe**: Definiere, welche Tiefe der Entscheidungsbaum maximal haben soll.

# %%
attribute  = ["Energie","Fett"]
zielkriterium = "Label"
baumtiefe = 2

# %% [markdown]
# ## Berechnung des Entscheidungsbaums
#

# %%
baum = sklearn.tree.DecisionTreeClassifier(max_depth=baumtiefe) # Erzeugen des Baums
baum.fit( daten[attribute], daten[zielkriterium] )              # Berechnung des Baums

# %% [markdown]
# ## Darstellung des Entscheidungsbaums
# Für die Darstellung des Baums können verschiedene Optionen gewählt werden. Eine [Dokumentation der Parameter kann hier aufgerufen werden.](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)   

# %%
sklearn.tree.plot_tree( baum, feature_names=attribute, label="none", filled=True,
                        class_names=["ungesund","gesund"], impurity=False,
                        proportion=False, fontsize=8 )
matplotlib.pyplot.show()

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 1
# - Verändere die Parameter **Attribute** und **Baumtiefe** und beobachte die Veränderungen im Entscheidungsbaum.
# - Verändere die Darstellung des Entscheidungsbaums.
#

# %% [markdown]
# # Kapitel 3: Wie gut ist meine KI?
# Die Daten der Lebensmittel aus der Berechnung (die Trainingsdaten) kann man nun in einem ersten Schritt zur Kontrolle verwenden, ob der berechnete Entscheidungsbaum gut funktioniert. Dazu testen wir für die einzelnen Datensätze nacheinander, ob sie mit dem Entscheidungsbaum richtig zugeordnet werden. Dies geschieht mit der Funktion **predict**.
#
# ## Die Funktion predict
# Die Funktion bekommt einen Datensatz übergeben und gibt die Zuordnung zurück, die sich aus dem trainierten Entscheidungsbaum ergibt.

# %%
korrekt = 0
falsch = 0
for datensatz in daten.index:  #Schleife über alle Datensätze
    if( baum.predict(daten.loc[[datensatz]][attribute]) == (daten.loc[datensatz]['Label']) ):  # Vorhersage ist wie im Label hinterlegt?
        korrekt += 1
    else:
        falsch += 1
        print("Fehler bei", daten.loc[datensatz]['Name'])
gesamt = korrekt + falsch
print( "Anzahl falsch klassifiziert: ",  falsch, "(" , round(falsch/gesamt*100,2), "% )")  
print( "Anzahl korrekt klassifiziert: ", korrekt, "(" , round(korrekt/gesamt*100,2), "% )")  

# %% [markdown]
# ## Verbesserung des Modells

# %% [markdown]
# <div style="padding: 5px; border: 5px solid #0077b6;">
#
# ### Aufgabe 2
#
# Durchlaufe für die falsch eingeordneten Lebensmittel den Entscheidungsbaum oben von Hand und analysiere, warum sie falsch klassifiziert wurden. Passe anschließend die Parameter der Berechnung gezielt an, um die Fehlerzahl zu minimieren.
#
#

# %% [markdown]
# ---
# # Wie geht es weiter?
# Du hast deinen Entscheidungsbaum auf dieser Seite für eine fest vorgegebene Zahl von Trainingsdaten optimiert. Das Ergebnis hängt dabei aber stark von den gewählten Beispielen ab. Getestet wurde der Baum zudem nur mit den Trainingsdaten.  
#
# Im nächsten Schritt soll daher bei der Berechnung des Entscheidungsbaums die Auswahl und die Anzahl der Trainings- und Testdaten verändert werden können. Öffne dazu das Notebook im nächsten Kapitel...

# %% [markdown]
#
