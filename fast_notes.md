Bei bike_shring_demand geht es darum, auf Basis von Daten wie Temp., Jahreszeit, Uhrzeit ... die Anzahl Fahrradverleihe in dieser Stunde zu predicten
Bei brazilian_houeses geht es darum, auf Basis von Daten wie Flächer, Zimmer, ... den Gesamtpreis der Immo in log-Skala zu predicten
Bei superconductivity geht es darum, auf Basis von Daten wie Atomarer Masse, Schmelztemp. ... zu predictend wann das Material supraleitend wird
Bei wine_quality geht es darum, auf Basis von Daten wie Zucker-/Salzgehalt, Alk-gehalt ... die mittlere Wein Qualität zu predicten

Bei yprop_4_1 geht es darum, auf Basis von über 250 numerischen Merkmalen (oz1–oz251), die vermutlich chemische, physikalische oder statistische Eigenschaften repräsentieren, den Zielwert oz252 vorherzusagen – ein kontinuierlicher Wert zwischen 0 und 1, der vermutlich eine proportionale Größe, Wahrscheinlichkeit oder Materialeigenschaft beschreibt.

# Die Ordner 1-10 innerhalb der Datasets sind CV Folds die uns schon vorgegeben wurden, also ein Datensatz auf 10 verschieden Train/Test Splits aufgeteilt.
# Innerhalb der Folds, also z.b. in Ordner 1 gibt es dann den Train/Test Split. X_train sind die Trainingsdatensätze, y_train das zugehörige Label in einer eigenen Datei. Da der Split ja vorgegeben ist, sind in X_test und y_test die Testdatensätze, auf denen ich das Modell testen soll, auch mit Label in y_test.

# Dass manchmal ein Fold eine Zeile mehr hat als der andere liegt daran, dass die Gesamtzahl an Daten nicht immer durch 10 Teilbar ist, bike_sharing_demand ist mit 7820 Zeilen z.b. perfekt Teilbar => jeder Fold hat exakt 782 Zeilen

### AutoML Pipeline sollte möglichst kategorische von numerischen Features unterscheiden können, was nicht so trivial ist. Z.b. ist "hour" also die Uhrzeit beim bike_sharing_demand zwar kategorisch, aber mit int8 werten, also Nummern geschrieben, das macht es erstmal nicht möglich direkt zu sehen dass das kategorisch ist, vor allem bei AutoML ohne Datensatzkenntnis oder anonymen Featurenamen   -   am besten sowohl Datentyp als auch unique/samples anschauen. Wenn float oft numerisch, int oft kategorisch. Wenn int und wenig uniques rel. sicher kategorisch, wenn float und viele uniques rel. sicher numerisch   -   GPT sagt sogar mit sin() verrechnen, weil dann dem Modell der Zusammenhang von 0 und z.b. 23 klar wird, was bei reiner category nicht der Fall wäre, aber auch overkill vllt.

# Alle Datensätze haben keine Missing Values, nicht ein einziges Feature

# Evtl. krasse skews oder ausreißer handlen, aber gbdts können i.d.R. gut damit umgehen, also nur bei ultra krassen dingern, dürfte schnell implementiert sein und keine performance kosten

# Wie wollen wir Arbeit teilen ? Alle machen alles = geil fürs Team, explizit aufsplitten kann integration schwerer machen,aber zeitlich viel effizienter
# Am besten so dass ich jetzt so viel wie möglich machen kann

# Bei pipelineprogrammierugn aufpassen, dass ich variablen wieder lösche damit der ram nicht explodiert


# zum Feature Engineering keine DFS, weil wir nur eine Tabelle haben, ohne Entities wie UserID oder so, zu der dann mehrere Spalten gehören, es ist also keine Aggregation über mehrere Zeilen mehr nötig. Es gibt ja außerdem keine Zeitreihen, Historien oder so wie z.b. mehrere Einkäufe eines Kunden, also wenig sinnvoll

# Vllt. implementieren, dass ich bei einem gewissen R2 von z.b. >0.9 oder so gar kein FE mache, weil das bisher nur noise oder extrem geringe verbesserungen gebracht hat
# so z.b.
if mean_cv_r2 > 0.91:
    skip_feature_engineering()
elif mean_cv_r2 > 0.85:
    apply_light_feature_engineering()
else:
    apply_full_feature_engineering()

# genetic featureengineering war für die schlechten r2s stark

# Die Datensätze sind nicht wie bei K-Fold CV geteilt, sondern 10 zufällige Test Train Splits auf dem gesamten Dataset. Also enthält ein Fold auch immer alle Trainings / Testdaten, nur komplett zufällig, nicht wie bei einer 10 Fold CV




Wichtige Fragen, die wir noch von den Prüfern beantwortet bekommen sollten.

Inwieweit zählt nur der R2 Score für die Note und sind auch andere Pipeline-spezifische Metriken wichtig wie Speichereffizienz, Anzahl genutzter Features, Prediction / Trainingsgeschwindigkeit, **generell wie lange die Pipeline braucht, oder sollen wir die vollen 24h gut versuchen auszunutzen, wenn möglich ?**

Soll das ganze Projekt generell für quasi x-beliebige Tab-ML Daten als Regression geeignet sein, oder darf es auch Datensatz spezifischer sein ? Z.B. wenn der finale Trainingsdatensatz wenig Ausreißer hat, müssen wir dann trotzdem ein Ausreißer-Handling einbauen, weil das ja für andere Regressionsaufgaben / -daten wichtig sein könnte ? Auch z.b. bei der HPO, dürfen wir die auf den finalen Datensatz anpassen, oder ist das Ziel aus jedem x-beliebigen Datensatz eine HPO von Grund auf laufen zu lassen ?

Wie sieht es mit der Verwendung von Advanced Methoden aus ? Ist das ein absolut wichtiges Kriterium dass wir da welche reinpacken, auch wenn das vllt. nicht ganz so sinnvoll für die Task ist, oder nur machen wenn es absolut Sinn macht ? Z.b. bei so kleinen Datasets für TabML wie wir haben, macht für die HPO eine Optuna TPE mit Hyperband / Medianpruner und earlystopping ja viel mehr Sinn (weil die Modelle so schnell trainiert werden können), als Bayesian Optimization, aber das wäre halt mehr advanced

Was für Rechenressourcen darf man für den Stichtag annehmen, wenn unsere Pipeline getestet wird ? RAM, CPU ...

Die Frage mit Cat Num Handling