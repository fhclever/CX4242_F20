# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import tree
from sklearn.tree import _tree
import networkx as nx
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4
from bokeh.io import output_file, show
from bokeh.layouts import column, row
from bokeh.models import Div, ColumnDataSource, CustomJS, Dropdown, Select, DataTable, DateFormatter, TableColumn
from bokeh.palettes import Spectral6, Category20
from bokeh.plotting import figure
import joblib
from bokeh.transform import factor_cmap
from bokeh.io import curdoc
np.random.seed(4)


def analyze(df):
    X1 = df.iloc[:, 0:7]
    X2 = df.iloc[:, 8:]
    y = df["Cell_class"]
    new_X = pd.concat([X1, X2], axis=1)
    data = new_X
    rforest = joblib.load('rForestAlgorithm.joblib')
    labels = ['Astrocyte', 'Inhibitory', 'OD Mature 2', 'Endothelial 1', 'Ambiguous',
              'Pericytes', 'Endothelial 2', 'OD Mature 1', 'OD Immature 1', 'Excitatory',
              'Microglia', 'Endothelial 3', 'OD Mature 4', 'OD Immature 2', 'OD Mature 3',
              'Ependymal']
    train_data = data.fillna(0)
    X = train_data.drop(
        ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Behavior', 'Bregma', 'Centroid_X', 'Centroid_Y', 'Neuron_cluster_ID'],
        axis=1)
    xaxisvals = []
    for col in X.columns:
        xaxisvals.append(col)
    yaxisvals = []
    for col in X.columns:
        yaxisvals.append(col)
    y_predict = rforest.predict(X).tolist()

    palette = Category20[16]
    X["label"] = y_predict

    Columns = [TableColumn(field=Ci, title=Ci) for Ci in X.columns]  # bokeh columns
    bTable = DataTable(columns=Columns, source=ColumnDataSource(X))  # bokeh table

    #X.to_csv("RandomForestResults.csv")
    X["color"] = 0
    # print(X.iloc[:,-1])
    #print(len(labels))
    for i in range(X.shape[0]):
        for j in range(len(labels)):
            if X.iloc[i, -2] == labels[j]:
                X.iloc[i, -1] = palette[j]
        # colorval = np.where(labels == X.iloc[i,-1])
        # print(colorval)
    #print(X.columns)
    source = ColumnDataSource(data=X)
    #print(source)
    div = Div(text="Random Forest Algorithm Prediction (Accuracy of 82.6%)", width=400, height=100, style={'font-size': '200%'})
    xselect = Select(title="Select X Axis:", value=xaxisvals[4], options=xaxisvals)

    # xcallback = CustomJS(code="console.log('xdropdown: ' + this.item, this.toString())")
    # xselect.js_on_change("value", xcallback)

    yselect = Select(title="Select Y Axis:", value=yaxisvals[0], options=yaxisvals)

    # ycallback = CustomJS(code="console.log('ydropdown: ' + this.item, this.toString())")
    # yselect.js_on_change("value", ycallback)
    p = figure(plot_width=800, plot_height=400, sizing_mode="scale_both",
               y_axis_label=yselect.value, x_axis_label=xselect.value)
    p.circle(x=xselect.value, y=yselect.value, source=source, size=7, color="color", line_color=None, legend_group="label")

    div2 = Div(text = "Here, we display the results of the Random Forest classifier. You can select two genes of interest to have them plotted on the X and Y axes and the resulting points will be color coded according to the legend based on the model predictions.",
               width = 400,
               height = 200)

    col1 = column(div, xselect, yselect, div2)

    text2 = Div(
        text="Table of input values. The final column reveals the prediction from this classification algorithm.",
        width=100, height=100)
    tableRow = row(bTable, text2)

    layout = row(col1, p)

    def update():
        global p
        xaxis = xselect.value
        yaxis = yselect.value
        p = figure(plot_width=800, plot_height=400,
                   x_axis_label=xaxis, y_axis_label=yaxis,
                   sizing_mode="scale_both")

        p.circle(x=xselect.value, y=yselect.value, source=source,
                 size=7, color="color", line_color=None, legend_group="label")

        layout.children[1] = p

    def test_function(attr, old, new):
        update()

    xselect.on_change("value", test_function)
    yselect.on_change("value", test_function)
    outLayout = column(layout, tableRow)
    return outLayout, y_predict
