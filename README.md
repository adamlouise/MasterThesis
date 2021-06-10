# MasterThesis
Tri des codes utiles pour le mémoire

J'ai mis en *italique* les codes qui je pense t'intéresseront le plus pour l'article 


### Les codes principaux pour entrainer et tester les modèles:

- *Network1_usingDico_v2.py*: purement DL

- *Network2_bigversion.py*: DL après NNLS

- DecisionTrees.py: (purement) RF & GB

- M1.py: Exhaustive search (je changerais bien le nom - qui est assez perturbant - mais une fois que le fichier est chargé sur github je n'arrive pas à l'enlever ou à le renommer)
    - résolution du exhaustive search (avec true/estimated orientations en fonction de la ligne 155/156)


### Classes: 
uniquement utilisées pour load les modèles dans les codes de comparaison, une copie est aussi présente directement dans les codes ici au dessus (--> encore à changer)

- Net1_Class.py: classe correspondant au Network1
- Net2_Class.py: classe correspondant au Network2


### Codes de test (et graphes):

- ComparingTests.py: compare les 4 méthodes
- *ComparingTests_Descaled.py*: pareil mais avec quelques légers changements pour avoir des erreurs en "vraies unités"


### Générer les données:

- *gen_synthetic_data_new2_gaetan.py*: (suite au code dont on avait discuté fin mai) 
    - utilisé pour générer mes données d'entrainement (600 000)
    - orientations ESTIMEES (pas les vraies)

- gen_synthetic_data_new2_gaetan_TEST.py: pareil que le précédent mais adapté pour données de test
    - utilisé pour générer les données de test (15 000), SNR et nu fixés
    - orientations ESTIMEES 

- getDataW.py: code qui fait les manips pour transformer les vecteurs de poids w sparse en vecteurs utilisables (inspiré d'un code que tu m'avais envoyé en début d'année, pas besoin si les w sont stockés de manière non sparse)
