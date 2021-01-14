import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score
from bokeh.io import output_file, show
from bokeh.models import Select, Div, BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.layouts import column, row
from bokeh.palettes import cividis, Category20
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import pickle

def create_model(df):
    train_size = 1  # subject to change, I set it to 27848 to run the classifer faster for demo purpose only
    train_data = df.sample(frac = train_size)
    train_data = train_data.fillna(0)
    y = train_data[["Behavior", "Cell_class"]]
    X = train_data.drop(
        ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Behavior', 'Bregma', 'Centroid_X', 'Centroid_Y', 'Cell_class',
         'Neuron_cluster_ID'], axis=1)
    #print(y)

    # One hot encoding

    uniqueBehaviors = y['Behavior'].unique()
    y_new = np.zeros((len(y), len(uniqueBehaviors)))

    idx = 0
    for i, j in y.iterrows():
        index = uniqueBehaviors.tolist().index(j['Behavior'])
        y_new[idx, index] = 1
        idx = idx + 1

    y_new = pd.DataFrame(data=y_new, columns=uniqueBehaviors)
    #print(y_new)

    X_train, X_test, y_train, y_test = train_test_split(X, y_new, train_size=0.7, random_state=1)
    X_train.head()
    #print(y_train)



    models = []
    #print(X_test.shape)
    #print(y_test.shape)
    def predict(X_test, y_test, clfs=models, uB=uniqueBehaviors):
        out_predict = []
        numData = X_test.shape[0]
        for i in range(numData):
            out_predict = out_predict + ['nan']


        bIdx = 0
        precisions = []
        recalls = []
        conds = []
        for clf in clfs:
            prediction = clf.predict(X_test)
            precision = precision_score(y_test[bv], prediction)
            precisions = precisions + [precision]
            recall = recall_score(y_test[bv], prediction)
            recalls = recalls + [recall]
            cond_count = clf.ruleset_.count_conds()
            conds = conds + [cond_count]
            idx = 0
            for pred in prediction:
                if pred:
                    out_predict[idx] = uB[bIdx]
                idx = idx+1

            bIdx = bIdx + 1
        return out_predict, precisions, recalls, conds

    for bv in uniqueBehaviors:
        clf = lw.RIPPER(k=2, dl_allowance=4)
        clf.fit(X_train, y_train[bv], pos_class=1)
        models = models + [clf]

    (out_predict, precisions, recalls, conds) = predict(X_test, y_test, models, uniqueBehaviors)
    #print(out_predict)
    #print(precisions)

    #print(type(uniqueBehaviors))
    uniqueBehaviors = uniqueBehaviors.tolist()
    if 'nan' not in uniqueBehaviors:
        uniqueBehaviors = uniqueBehaviors + ['nan']

    matrix = np.zeros((len(uniqueBehaviors), len(uniqueBehaviors)))
    #print(uniqueBehaviors)
    #print(len(out_predict))
    idx = 0
    for index, rows in y_test.iterrows():
        rIdx = rows.argmax()
        #print(rIdx)
        #print(rows)

        predic = out_predict[idx]
        cIdx = uniqueBehaviors.index(predic)
        #print(cIdx)
        matrix[rIdx, cIdx] = matrix[rIdx, cIdx] + 1
        idx = idx + 1

    outMatrix = pd.DataFrame(data = matrix, index = uniqueBehaviors, columns = uniqueBehaviors)
    outMatrix.index.name = 'Actual'
    outMatrix.columns.name = 'Predicted'
    #print(outMatrix)
    pickle.dump(outMatrix,open('rippeMatrix.p', 'wb'))
    pickle.dump(models,open('ripperModel.p', 'wb'))
    return outMatrix

def analyze(df):
    X1 = df.iloc[:, 0:7]
    X2 = df.iloc[:, 8:]
    y = df["Cell_class"]
    new_X = pd.concat([X1, X2], axis=1)
    #print(new_X.shape)
    #print('AAAAAAAA')
    data = new_X
    labels = ['Astrocyte', 'Inhibitory', 'OD Mature 2', 'Endothelial 1', 'Ambiguous',
              'Pericytes', 'Endothelial 2', 'OD Mature 1', 'OD Immature 1', 'Excitatory',
              'Microglia', 'Endothelial 3', 'OD Mature 4', 'OD Immature 2', 'OD Mature 3',
              'Ependymal']
    train_data = data.fillna(0)
    #print(train_data.shape)
    #print('BBBBBBBB')
    X = train_data.drop(
        ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Behavior', 'Bregma', 'Centroid_X', 'Centroid_Y', 'Neuron_cluster_ID'],
        axis=1)
    xaxisvals = []
    for col in X.columns:
        xaxisvals.append(col)
    yaxisvals = []
    for col in X.columns:
        yaxisvals.append(col)

    models = pickle.load(open('ripperModel.p', 'rb'))
    uniqueBehaviors = pickle.load(open('ripperUB.p', 'rb'))
    labels = uniqueBehaviors
    def predict(X_test, clfs=models, uB=uniqueBehaviors):
        out_predict = []
        numData = X_test.shape[0]
        for i in range(numData):
            out_predict = out_predict + ['nan']


        bIdx = 0
        precisions = []
        recalls = []
        conds = []
        for clf in clfs:
            prediction = clf.predict(X_test)
            idx = 0
            for pred in prediction:
                if pred:
                    out_predict[idx] = uB[bIdx]
                idx = idx+1

            bIdx = bIdx + 1
        return out_predict
    y_predict = predict(X,clfs = models,uB=uniqueBehaviors)
    palette = Category20[16]
    X["label"] = y_predict
    #X.to_csv("RandomForestResults.csv")

    Columns = [TableColumn(field=Ci, title=Ci) for Ci in X.columns]  # bokeh columns
    bTable = DataTable(columns=Columns, source=ColumnDataSource(X))  # bokeh table
    text2 = Div(
        text="Table of input values. The final column reveals the prediction from this classification algorithm.",
        width=100, height=100)
    tableRow = row(bTable, text2)

    X["color"] = 0
    # print(X.iloc[:,-1])
    #print(len(labels))
    for i in range(X.shape[0]):
        for j in range(len(labels)):
            if X.iloc[i, -2] == labels[j]:
                X.iloc[i, -1] = palette[j]
        # colorval = np.where(labels == X.iloc[i,-1])
        # print(colorval)
    #print(X)

    source = ColumnDataSource(data=X)
    #print(source)
    div = Div(text="RIPPER Algorithm Prediction (Accuracy of 68.52%)", width=400, height=100,
              style={'font-size': '200%'})
    xselect = Select(title="Select X Axis:", value=xaxisvals[4], options=xaxisvals)
    yselect = Select(title="Select Y Axis:", value=yaxisvals[0], options=yaxisvals)

    div2 = Div(
        text="Here, we display the results of the RIPPER classifier. You can select two genes of interest to have them plotted on the X and Y axes and the resulting points will be color coded according to the legend based on the model predictions.",
        width=400,
        height=200)

    p = figure(plot_width=800, plot_height=400, sizing_mode="scale_both",
               y_axis_label=yselect.value, x_axis_label=xselect.value)
    p.circle(x=xselect.value, y=yselect.value, source=source, size=7, color="color", line_color=None,
             legend_group="label")
    col1 = column(div, xselect, yselect, div2)
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


    matrix = pickle.load(open('ripperClass.p', 'rb'))
    colors = cividis(256)

    matrix = matrix.stack().rename("value").reset_index()

    # Had a specific mapper to map color with value
    mapper = LinearColorMapper(
        palette=colors, low=matrix.value.min(), high=matrix.value.max())



    f = figure(plot_height=400,
               plot_width=800,
               title="Confusion Matrix",
               x_range=list(matrix.Actual.drop_duplicates()),
               y_range=list(matrix.Predicted.drop_duplicates()),
               toolbar_location=None,
               tools="")
    f.rect(
        x='Actual',
        y='Predicted',
        width=1,
        height=1,
        source=ColumnDataSource(matrix),
        line_color=None,
        fill_color=transform('value', mapper))

    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        ticker=BasicTicker(desired_num_ticks=10))

    f.add_layout(color_bar, 'right')

    text = Div(
        text="This confusion matrix visualizes the predictive accuracy of this model for each individual feature.",
        width=400, height=400)
    row2 = row(f, text)

    outFig = column(layout,row2,tableRow)
    return outFig, y_predict






