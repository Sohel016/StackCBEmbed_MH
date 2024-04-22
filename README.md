# StackCBEmbed_MH
This contains the scripts used during metaheuristic term project.
The Training folder contains the training code.  The prediction folder contains the prediction code. 
The hyperparameter tuning and  feature selection codes are in the folders with same name.
You need to download the required csv files in the directories for running the scripts properly. See the readme files in the directories for which files to download and more clearcut instructions.
The FeatureSelection codes will output a csv file , containing which features were selected. They are number serially 0-1023 being Protein LM embeddings, 1024-1043 PSSMs, 1044 monogram , 1045-1444 DPC, 1445 ASA,1446-1447 HSE,1448-1449 Torsion angles, 1450-1456 Physiochemical properties. You can then train the training script based on selected features. The training will generate some .sav files, which are the trained models, place them into the directory of prediction file for making the prediction.
