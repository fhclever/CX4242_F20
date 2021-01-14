from bokeh.plotting import figure, output_file, show, save, ColumnDataSource
from bokeh.transform import dodge
import gene_expression_analysis
import marker_gene_visualization
import rf_viz
import ripper_Classification
import neuralNetwork
from bokeh.io import curdoc
import hClustering
import helper
from bokeh.transform import dodge
import random
from bokeh.layouts import column, layout, row
from bokeh.models import CustomJS, MultiChoice, RadioButtonGroup, Div, Select, FileInput, Panel, Tabs
import pandas as pd
########################################################################################################################
# Data Import
file_name = 'Sample_Data.csv'
ogdf = pd.read_csv(file_name, sep=',', nrows=200)

#file_name1 = 'data.csv'
#bigDf = pd.read_csv(file_name1, sep=',', skiprows=lambda i:i>0 and random.random() > 0.025)
#print("data read")
#print(bigDf.shape[0])
#outMatrix = ripper_Classification.create_model(bigDf)
#stop

file_name2 = 'Sample_Data.csv'
df2 = pd.read_csv(file_name2, nrows=60)

print("Data imported")
#print(ogdf.shape[0])

file_input = FileInput(accept = '.csv')
div = Div(text="""
        Welcome to MARC, the Merfish Analysis for Real-time Classification tool! Here you can upload your MERFISH data 
        and analyze gene expression through descriptive statistics, hierarchical clustering, hierarchical latent tree 
        analysis (HLTA), and machine learning classification models, including RIPPER, random forest, and neural 
        networks.\n\n

        Please ensure that the file you are uploading is located in the same folder as main.py.\n\n

        Classification output will be generated in the same folder as 'your_file_input_ANALYSIS.csv'.
    """)
WelcomeCol = column(div, file_input)
tab0 = Panel(child=WelcomeCol,title="Welcome")
tab1, tab2, tab3, tab4 = None, None, None, None


def update(data,fn):
	global tab1, tab2, tab3, tab4
	global allTabs

	########################################################################################################################
	# Panel 1: Gene Expression Analysis
	GEAcol = gene_expression_analysis.analyze(data)
	tab1 = Panel(child=GEAcol, title="Gene Expression Analysis")
	print("Panel 1 created")
	#print(len(data))
	########################################################################################################################
	# Panel 2: Clustering

	HClusCol = hClustering.analyze(data)
	tab2 = Panel(child=HClusCol, title="Hierarchical Clustering")
	print("Panel 2 created")
	#print(len(data))
	########################################################################################################################
	# Panel 3: Marker Gene Analysis

	MGVLayout = marker_gene_visualization.analyze(data)
	tab3 = Panel(child=MGVLayout, title="Marker Gene Analysis")
	print("Panel 3 created")
	#print(len(data))
	########################################################################################################################
	# Panel 4: Classification - NN, Random Forest, RIPPER


	#outMatrix = ripper_Classification.create_model(data)
	(RipperVis, ripper_predict) = ripper_Classification.analyze(data)
	subTab1 = Panel(child=RipperVis, title='RIPPER Classification')
	print("Ripper layout created")

	(NNCol, NN_predict) = neuralNetwork.analyze(helper.drop_empty_rows(helper.genes_only(data)))
	subTab2 = Panel(child=NNCol, title='Neural Network Classification')
	print("Neural Network layout created")

	#print(len(NN_predict))
	#print(len(ripper_predict))
	#print(data)

	(RFLayout, RF_predict) = rf_viz.analyze(data)
	subTab3 = Panel(child=RFLayout, title="Random Forrest Classification")

	classTabs = Tabs(tabs=[subTab1, subTab2, subTab3])

	tab4 = Panel(child=classTabs, title="Classification")
	print("Panel 4 created")

	data['Random Forrest Prediction'] = RF_predict
	data['Neural Network Prediction'] = NN_predict
	data['RIPPER Prediction'] = ripper_predict
	data.to_csv(fn[:-4] + "_ANALYSIS.csv")

	allTabs.tabs[1] = tab1
	allTabs.tabs[2] = tab2
	allTabs.tabs[3] = tab3
	allTabs.tabs[4] = tab4
	print("Tabs recombined")


allTabs = Tabs(tabs = [tab0, tab1, tab2, tab3, tab4])
print("Tabs combined")

def file_change(attr, old, new):
	WelcomeCol = column(div, file_input, Div(text = str(file_input.filename) + " uploaded"))
	print("Processing file upload for file " + str(file_input.filename))
	allTabs.tabs[0] = Panel(child=WelcomeCol,title="Welcome to MARC")
	df3 = pd.read_csv(file_input.filename, sep = ',')
	update(df3,file_input.filename)

file_input.on_change('value',file_change)

update(ogdf, file_name)

#print(allTabs.tabs)

curdoc().add_root(allTabs)
#print(dir(curdoc()))
curdoc().title = "MARC"
print("Document created")