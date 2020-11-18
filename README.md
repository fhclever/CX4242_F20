# README

## DESCRIPTION

This program should be used to analyze gene expression data generated using the MERFISH technique and analysis found at https://github.com/ZhuangLab/MERFISH_analysis. Once
this has been done, the user can then choose to perform hierarichal latent tree analysis (HLTA), clustering, the RIPPER algorithm, the random forest algorithm, a neural
network, or gene expression analysis. We provide a sample data set, but the full data set (1GB) is available here: https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248.


## INSTALLATION

Dependencies: Python >=3.7, bokeh >=2.2.3, joblib >=0.15.1, matplotlib >=1.1.3, networkx >=2.5, numpy >=1.17.2, pandas >=1.1.3, pickle >=0.7.5, scipy >=1.5.2, 
seaborn >=0.9.0, sklearn >=0.0, wittgensen >=0.2.3.

Steps: Download my_gene_analyzer.zip and extract into my_gene_analyzer directory.


## EXECUTION

From the my_gene_analyzer directory, use the linux command:
     bokeh serve --show main.py
     
Upon opening the program, a sample data set will already be loaded, but there is a home page where you may upload your own data to look at. The data should have been
processed through the Zhuang lab MERFISH analysis (https://github.com/ZhuangLab/MERFISH_analysis) first, producing a file formated as bellow:

<pre>
+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+----------+--------+----------+
| Cell_ID | Animal_ID | Animal_sex | Behavior         | Bregma | Centroid_X | Centroid_Y | Cell_class  | Neuron_cluster_ID | Gene 1   | Gene 2 | Gene 3   |
+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+----------+--------+----------+
| Cell 1  | 1         | Female     | Naive            | 0.26   | -3211.56   | 2608.541   | Astrocyte   |                   | 1.638275 | 0      | 0.583729 |
+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+----------+--------+----------+
| Cell 2  | 1         | Female     | Parenting        | 0.26   | -3207.92   | 2621.795   | Inhibitory  | I-5               | 0        | 0      | 0.938416 |
+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+----------+--------+----------+
| Cell 3  | 2         | Male       | Virgin Parenting | 0.21   | 2045.93    | 3445.059   | OD Mature 2 | 2                 | 1.845902 | 0      | 1.384637 |
+---------+-----------+------------+------------------+--------+------------+------------+-------------+-------------------+----------+--------+----------+
</pre>

The user can choose to either demo the program with the sample data or upload their own. There are tabs available in the program for each of our analyses. Within each
tab, the user may select their desired parameters. The classification programs will perform predictions on the cell class of the user's input and will then produce
an output file that will be stored in the my_gene_analyzer/data directory.


## DEMO VIDEO
The following video briefly details installation and setup.
