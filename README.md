# Malware-DGA-detection

DBSCANclusting, Domain feature coding, weka machine learning classifying.

This machine learning based Domain classifier can classify Domain Generation Algorithms(DGAs) generated domains from input.

The main class Dictionaryvalidwords contains a domain feature extractor, arff file writer, and weka machine learning training model.

Every domain will be process by a feature extractor and the result could be classified by weka machine learning predict model based on the training model.

The Dbscanfunction implements a Density-based spatial clustering of applications with noise (DBSCAN) clustering algorithm based on the features we got in main class. The clustering result is used for further malware DGA domain prediction. A Deep learning model for DGA is developed in SoftmaxfoDGA.py

