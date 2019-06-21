Classification and Analysis of Mushroom Dataset
==================================================

Abstract
=======

Aim to look at the illustration and documentation of how the classification of the mushroom dataset can be used as an aid to differentiate poisonous and edible mushrooms, as well as analysing the different classifiers.

Introduction
=======
Due to the evolving of computer science and the fast development and vast usage of World Wide Web and other electronic data, information extraction is a popular research field. Data mining is the process of analysing data from different perspective and summarizing it into useful information. The main aim of data mining is to uncover relationship in data and predict the outcome. Data mining involves many methods such as clustering, classification, summarization and sequential pattern matching. 

Dataset
=======
This database contains 22 attributes and 8124 instances. It is 374KB and is freely available on the internet, on the UCI Machine Learning Repository. This dataset contains descriptions of hypothetical samples corresponding to 23 of gilled mushrooms. The “class” field refers to the edibility of the mushroom, i.e. whether the species is identified as edible or poisonous. This is marked as “p” for poisonous or “e” for edible. The figure below shows the other attributes present in this dataset:
![image](https://user-images.githubusercontent.com/25173957/59938437-deecd880-944c-11e9-8e3f-71dfea6caa9d.png)



Background
==========

There are 22 attributes to be considered in this dataset and each one of them describes a particular aspect of the mushroom. A mushroom is the construction of a fungus, normally situated above the ground on soil scavenging on nutrients in research of food source.

To identify a mushroom can be a quite tricky situation as you cannot tell as first close if one is edible or not or poisonous from non-poisonous. As well as their shape and colour would also help to identify their nature. The spore of a mushroom named basidiospores,produced on the gills allows them to produce a rain of powder under the cap. In a mushroom there are different attributes that needs to be considered in order the structure of such.

The cap is the skeleton which is based at the top of the mushroom meaning is correlates to the head of the mushroom. The cup(volva), base of the mushroom is the pre-existing base of the mushroom before developing into a mature mushroom. This can be found in some, not all have it. Gills are arranged at the centre located at the lower-side of the mushroom cup. The spores are born and produced from the gills.

The ring of the mushroom is the remaining of the velvet (this is the tissue that allows the connection of stem and the cap before the mushroom develops into its new body. Some mushroom would have a feature that can only be seen and analysed when cut or either bruised this is when their original colour would drastically change pigment, by doing so it can be analysed and be able to be classified depending on its colour and be identified. A feature to be considered when describing a mushroom is their veil, a mushroom has 2 different types of veil: a universal veil and a partial veil.

The universal veil is the premature tissue that constructs the mushroom before becoming a matured and fully-grown mushroom. The process of such would be due and the tissue would rupture and be slowly dissolve, by doing so the mushroom would expand and mature gradually, however as we humans have a scar from our birth which is the scar from the incision of the belly button a mushroom would a similar aspect and would have traces of the veil within itself.

The partial veil or inner veil consists of a temporary tissue found in some mushrooms. The role of such is to protect the body of the fungus while its producing the section which will later be used to produce the spores, this is found on the lower surface of the mushroom cap.

Conclusion
=========== 

In this research, the three classification techniques such as decision tree, logistic regression and Naïve Bayes were used to evaluate the percentage of accuracy, precision, recall and F1 score for the mushroom dataset. Several modifications were done to the dataset before undergoing classification i.e. data cleaning and data transformation. Running the three classification models has shown that decision tree, as hypothesized, is the most efficient algorithm, compared to the others.

However, this did not mean that the results were valid because classification models were done on a specific subset of the data. Therefore, cross-validation was performed. Cross validation allowed me to analyse the machine learning models with a better confidence level. in the first case, we have an overfitting problem, which affected the accuracy of the models. Therefore, cross-validation improved the validity of the results which I have obtained.

As a possible future development, this present work can be extended in several directions, one such direction could be re-exploring these algorithms with different datasets and seeing how these compare with my current results or exploring different algorithms with the same datasets.

Limitations
==========
Because of the missing values, extensive data cleaning was required, which reduced the number of attributes, reducing the accuracy. However, had a different dataset been picked without missing values, this could have improved the accuracy of the classification models.

Another limitation is that there are no previous data on the edibility of mushrooms, which means that there is no comparison for this dataset, which could have been used in order to predict whether mushrooms are edible or poisonous.

*****FULL REPORT CAN BE FOUND IN THE PROJECT TO BE VIEWED******

*****copyright('Copyright (c) 2019, 'Lahiru Fernando');*****
