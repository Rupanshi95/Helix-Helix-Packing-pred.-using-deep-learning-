# Helix-Helix-Packing-pred.-using-deep-learning and regression

The computational description and prediction of HHPs has a long history, with first efforts dating back to the early 1980ies. Now, 50-years later, the massively increased dataset size of known protein structures (>100,000 in the PDB), and the arrival of new prediction architectures, call for revisiting the question of HHPs. According to the framework-model of protein folding, the packing of secondary structural elements (e.g. HHPs) has been regarded as an intermediated step during protein folding, a notion that remains valid to this day.

Approach in the project

Based on a representative set of PDB-structures (selected by using the program CD-Hit to render the set non-redundant), i well train a neural network to predict the packing geometry of HHPs (primarily the so-called packing dihedral angle), based on detected HHPs in proteins. For the identification of HHPs, i firstly used the helix software. Implicitly, i made the assumption that intra-protein HHPs generalize to the application of inter-protein HHP interactions.HHP-prediction can be treated as a classification (binned packing angles) or as
a regression problem (continuous angle prediction). I will use various architectures (MLP, CNN, LSTM).  In the process, i will examine the role of sequence-embeddings. I am aiming to determine the optimal embedding dimensionality (20 amino acid types are to be embedded into an N-dimensional space). This is not only of interest with regard to prediction performance, but may also shed light on the minimal number of amino acid types that “nature could have chosen”, a question that has frequently been discussed in the past.	
Recently, pre-trained amino-acid sequence-embeddings have become available, which i plan to test for suitability with regard to our prediction task.