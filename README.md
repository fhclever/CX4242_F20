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


DEMO VIDEO



The zip folder Faye made has a directory called "Code". Go to the directory right before that (so in this case go to the directory "Program"), and run the command:

bokeh serve --show Code

This will run a main.py file I have in there, which opens up my code. I think this will work pretty well once we do some more bokeh buttons in main.py. After that, we can just
continually call different python files. Only thing I haven't accounted for yet is how the user chooses which file they want to use. Right now it is hard coded into both my code
and Michael's.

Or is that something we could do for you?
Notes on final code: Don't think we really need that helper.py file. Also don't think we need matplotlib if we are directly using bokeh - other's should confirm.
