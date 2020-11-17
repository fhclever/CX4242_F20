README

DESCRIPTION

This program should be used to analyze gene expression data generated using the MERFISH technique and analysis found at https://github.com/ZhuangLab/MERFISH_analysis. Once
this has been done, the user can then choose to perform hierarichal latent tree analysis (HLTA), clustering, the RIPPER algorithm, the random forest algorithm, a neural
network, or gene expression analysis.


INSTALLATION

Dependencies: Python >=3.7, bokeh, joblib, matplotlib, networkx, numpy, pandas >=1.1.3, pickle, scipy >=1.5.2, seaborn >=0.9.0, sklearn, wittgensen.

Steps: Download my_gene_analyzer.zip and extract into my_gene_analyzer directory.


EXECUTION

From the my_gene_analyzer directory, use the linux command:
     bokeh serve --show main.py
     
Upon opening the program, a sample data set will already be loaded, but there is a home page where you may upload your own data to look at. The data should have been processed through the Zhuang lab MERFISH analysis (https://github.com/ZhuangLab/MERFISH_analysis) first, producing a file formated as bellow:

+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+--------+----------+----------+

| Cell_ID | Animal_ID | Animal_sex | Behavior         | Bregma | Centroid_X | Centroid_Y | Cell_class  | Neuron_cluster_ID | Gene 1 | Gene 2   | Gene 3   |

+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+--------+----------+----------+

| Cell 1  | 1         | Female     | Naive            | 0.26   | -3211.56   | 2608.541   | Astrocyte   |                   | 0      | 1.638275 | 21.29975 |

+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+--------+----------+----------+

| Cell 2  | 1         | Female     | Parenting        | 0.26   | -3207.92   | 2621.795   | Inhibitory  | I-5               | 0      | 0        | 1.578873 |

+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+--------+----------+----------+

| Cell 3  | 2         | Male       | Virgin Parenting | 0.21   | 2045.93    | 3445.059   | OD Mature 2 | 2                 | 0      | 1.845902 | 2.76886  |

+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+--------+----------+----------+


DEMO VIDEO
