#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import numpy as np
import scipy as sp
import math
import helper
from bokeh.io import output_file, show
from bokeh.layouts import gridplot, row, column
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter,
                          TextAreaInput, Paragraph, CustomJS,
                          Select,CategoricalColorMapper, Div)
from bokeh.plotting import figure
from bokeh.palettes import cividis,Category20
from bokeh.transform import transform
from random import random, randint, uniform
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

# In[2]:


# Define a neuron
class Neuron:

    def __init__(self, activation, bias = 0):
        self.bias = bias
        self.a = activation
        self.weights = []
        self.neighbors = []

    def init_weights(self, input_size):
        for i in range(input_size):
            self.weights.append(uniform(-2,2))

    def get_output(self, inputs):
        if 'predicted' in inputs.index:
            inputs = inputs.drop('predicted')
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs.iloc[i] * self.weights[i]
        total += self.bias
        self.output = helper.sigmoid(float(total))
        return self.output

    def calc_error(self, output):
        return 0.5*(output - self.output)**2

    def calc_pd_E_totInput(self, expected_output):
        return self.calc_pd_E_output(expected_output) * self.calc_pd_totInput_input()

    def calc_pd_E_output(self, expected_output):
        out = -1*(expected_output - self.output)
        return out
    def calc_pd_totInput_input(self):
        out = self.output * (1-self.output)
        return out

    def calc_pd_totInput_weight(self, index):
        out = self.inputs.iloc[index]
        return out


# In[3]:


# Define a layer of neurons
class Layer():

    def __init__(self, layer_size, input_size):
        self.layer_size = layer_size
        self.input_size = input_size
        self.neurons = []
        for i in range(self.layer_size):
            self.neurons.append(Neuron(0))
            self.neurons[i].init_weights(input_size)

    def get_outputs(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.get_output(inputs))
        return pd.DataFrame(outputs)

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])


# In[4]:


# Define network setup
class Network():
    lr = 0.5

    def __init__(self, input_size, output_size, hidden_size=30):
        self.first_layer_size = input_size
        self.hidden_layer_size = hidden_size
        self.output_layer_size = output_size
        self.hidden_layer = Layer(self.hidden_layer_size, self.first_layer_size)
        self.output_layer = Layer(self.output_layer_size, self.hidden_layer_size)

    def get_outputs(self, inputs):
        primary_outputs = self.hidden_layer.get_outputs(inputs)
        return self.output_layer.get_outputs(primary_outputs)

    def get_all_outputs(self, inputs, distinct):
        outputs = []
        for i in inputs.index:
            out = self.get_outputs(inputs.loc[i])
            outID = out.idxmax()
            outputs.append(distinct[outID.loc[0]])
        return outputs

    def train(self, training_inputs, expected_outputs):
        for i in range(len(training_inputs)):
            inputs = training_inputs.iloc[i]
            outputs = expected_outputs.iloc[i]

            self.get_outputs(inputs)

            pd_E_totInput_output = [0] * self.output_layer_size
            for i in range(self.output_layer_size):
                pd_E_totInput_output[i] = self.output_layer.neurons[i].calc_pd_E_totInput(outputs.iloc[i])

            pd_E_totInput_hidden = [0] * self.hidden_layer_size
            for h in range(self.hidden_layer_size):

                d_E_output_hidden = 0
                for g in range(self.output_layer_size):
                    d_E_output_hidden += pd_E_totInput_output[g] * self.output_layer.neurons[g].weights[h]

                pd_E_totInput_hidden[h] = d_E_output_hidden * self.hidden_layer.neurons[h].calc_pd_totInput_input()

            for i in range(self.output_layer_size):
                for w in range(len(self.output_layer.neurons[i].weights)):
                    pd_E_weight = pd_E_totInput_output[i] * self.output_layer.neurons[i].calc_pd_totInput_weight(w)

                    self.output_layer.neurons[i].weights[w] -= self.lr * float(pd_E_weight)

            for h in range(self.hidden_layer_size):
                for w in range(len(self.hidden_layer.neurons[h].inputs)):
                    pd_E_weight = pd_E_totInput_hidden[h] * self.hidden_layer.neurons[h].calc_pd_totInput_weight(w)

                    self.hidden_layer.neurons[h].weights[w] -= self.lr * float(pd_E_weight)

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.first_layer_size))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def calculate_error(self, inputs, outputs):
        total_error = 0
        for i in range(len(inputs)):
            self.get_outputs(inputs.iloc[i])
            for j in range(len(outputs.iloc[i])):
                total_error += self.output_layer.neurons[j].calc_error(outputs.iloc[i, j])
        return total_error


# In[5]:


def runNN(df):
    # Import data

    number_cells = df.shape[0]

    # Set size
    train_size = int(.7 * number_cells)
    train_data = df.iloc[0:train_size]
    train_data = train_data.fillna(0)
    test_data = df.iloc[train_size:]
    test_output = test_data['Cell_class']
    test_input = helper.drop_empty_rows(helper.genes_only(test_data))

    # Training/testing sets
    y = train_data['Cell_class']
    y_distinct = df['Cell_class'].unique().tolist()
    Y = [0] * len(y_distinct)
    Y = [Y] * len(y)
    Y = pd.DataFrame(Y, columns=y_distinct)
    for j in range(len(Y)):
        i = y_distinct.index(y[j])
        Y.iloc[j, i] = 1
    X = helper.drop_empty_rows(helper.genes_only(train_data))

    # determining batch size
    factors = []
    for i in range(1, X.shape[0] + 1):
        if X.shape[0] % i == 0:
            factors.append(i)
    batch_size = min(factors, key=lambda x: abs(x - 50))
    print(batch_size)

    nNum = [int(((2 / 3) * X.shape[0]) + len(y_distinct)), X.shape[0] * 2]
    networks = []
    for num in nNum:
        print("Hidden Neuron Number:", num)
        n = Network(X.shape[1], len(y_distinct), num)
        network = {"nn": 0, 'accuracy': [], 'class_labels': y_distinct}
        count = 0
        bcount = 0
        for i in range(0, X.shape[0] * 20, int(X.shape[0] / batch_size)):
            start = i % X.shape[0]
            if start + (int(X.shape[0] / batch_size)) >= X.shape[0]:
                end = X.shape[0] - 1
            else:
                end = start + (int(X.shape[0] / batch_size) - 1)
            n.train(X.loc[start:end], Y.loc[start:end])
            bcount += 1
            print(bcount % (batch_size), end=" ")
            if end == X.shape[0] - 1:
                count += 1
                print('epoch', count)
                network['accuracy'].append({'train': accuracy_score(y, n.get_all_outputs(X, y_distinct)),
                                            'test': accuracy_score(test_output,
                                                                   n.get_all_outputs(test_input, y_distinct))})
        network['nn'] = n
        network['output'] = dict()
        actual = n.get_all_outputs(X, y_distinct)
        class_act = classification_report(y, actual)
        network['output']['train'] = {'out': actual, 'classification': class_act}
        print(class_act)
        test = n.get_all_outputs(test_input, y_distinct)
        class_test = classification_report(test_output, test)
        print(class_test)
        conf_matrix = confusion_matrix(test_output, test)
        print(conf_matrix)
        network['output']['test'] = {'out': test, 'classification': class_test, 'matrix': conf_matrix}
        networks.append(network)
    return networks


# In[6]:


# networks = runNN('Sample_Data.csv')
# pickle.dump(networks,open('network.p','wb'))


# In[7]:


def acc_plot(network, title):
    epoch = range(1, len(network['accuracy']) + 1)
    accuracies = {'train': [], 'test': [], }
    for i in epoch:
        accuracies['train'].append(network['accuracy'][i - 1]['train'])
        accuracies['test'].append(network['accuracy'][i - 1]['test'])
    plot = figure(plot_width=600, plot_height=400, title=title)
    plot.line(epoch, accuracies['train'], line_width=3, legend_label='train', line_color='blue')
    plot.line(epoch, accuracies['test'], line_width=3, legend_label='test', line_color='red')
    plot.xaxis.axis_label = "Epoch"
    plot.yaxis.axis_label = "Accuracy"
    plot.legend.location = 'bottom_right'
    return plot


# In[8]:


def conf_matrix(network):

    matrix = pd.DataFrame(network['output']['test']['matrix'])

    matrix.index.name = 'true'
    matrix.columns.name = 'predicted'
    matrix.rename(columns=lambda s: network['class_labels'][s], index=lambda s: network['class_labels'][s],
                  inplace=True)
    matrix = pd.DataFrame(matrix.stack(), columns=['count']).reset_index()

    color_map = LinearColorMapper(palette=cividis(256),
                                  low=matrix['count'].min(),
                                  high=matrix['count'].max())

    p = figure(plot_height=400,
               plot_width=800,
               title="Confusion Matrix",
               x_range=list(matrix.predicted.drop_duplicates()),
               y_range=list(matrix.true.drop_duplicates()))
    p.rect(x='predicted',
           y='true',
           width=1,
           height=1,
           source=ColumnDataSource(matrix),
           fill_color=transform('count', color_map))
    legend = ColorBar(color_mapper=color_map,
                      location=(0, 0),
                      ticker=BasicTicker(desired_num_ticks=10))
    p.add_layout(legend, 'right')
    p.xaxis.axis_label = "Predicted"
    p.yaxis.axis_label = "True"

    text = Div(text= "This confusion matrix visualizes the predictive accuracy of this model for each individual feature.", width=400, height=400)
    rows = row(p,text)
    return rows


# In[9]:


def text_input():
    message = Paragraph(text='No input yet.')

    def Change_handler(attr, old, new):
        message.text = new

    t_input = TextAreaInput(value='Enter Data Here',
                            title="Enter data to get category:",
                            callback=CustomJS.from_py_func(Change_handler))
    return message, t_input


# In[10]:

def pred_plot(data,network,acc):
    predicted = network['nn'].get_all_outputs(data,network['class_labels'])
    data['predicted'] = predicted
    color_map = CategoricalColorMapper(palette = Category20[len(network['class_labels'])],factors = network['class_labels'])
    genes = list(data.columns)
    source = ColumnDataSource(data = data)
    xax = Select(title="x-axis:", value = genes[0], options = genes)
    yax = Select(title = 'y-axis: ', value = genes[5], options = genes)
    def update(attr, old, new):
        global p
        p = figure(plot_width = 800, plot_height = 400, y_axis_label = yax.value, x_axis_label = xax.value)
        p.circle(source = source, color = transform('predicted',color_map),x = xax.value, y = yax.value, size = 7,legend = 'predicted')
        plot.children[1] = p
    p = figure(plot_width = 800, plot_height = 400, y_axis_label = yax.value, x_axis_label = xax.value)
    p.circle(source = source, color = transform('predicted',color_map),x = xax.value, y = yax.value, size = 7,legend = "predicted")
    accuracy = Div(text="Neural Network Prediction (Accuracy of " + acc + " %)", width=400, height=100,
                   style={'font-size': '200%'})
    div2 = Div(
        text="Here, we display the results of the Neural Network classifier. You can select two genes of interest to have them plotted on the X and Y axes and the resulting points will be color coded according to the legend based on the model predictions.",
        width=400,
        height=200)
    plot = row(column(accuracy,xax,yax,div2),p)
    xax.on_change("value", update)
    yax.on_change("value", update)
    return plot, predicted


def plots(data_p):
    net_load = pickle.load(open('network.p','rb'))
    network = 0
    if net_load[0]['accuracy'][-1]['test'] > net_load[1]['accuracy'][-1]['test']:
        network = net_load[0]
    else:
        network = net_load[1]
    (pred, predicted) = pred_plot(data_p,network,str(network['accuracy'][-1]['test']*100))
    acc = acc_plot(net_load[0],"Lower Hidden Neuron Number Accuracy")
    acc2 = acc_plot(net_load[1],"Higher Hidden Neuron Number Accuracy")
    conf = conf_matrix(network)
    #text, inp = text_input()
    data_p["label"] = predicted
    Columns = [TableColumn(field=Ci, title=Ci) for Ci in data_p.columns]  # bokeh columns
    bTable = DataTable(columns=Columns, source=ColumnDataSource(data_p))  # bokeh table
    text2 = Div(text = "Table of input values. The final column reveals the prediction from this classification algorithm.",width = 100, height = 100)
    tableRow = row(bTable,text2)
    r = column([pred,row(acc,acc2),conf,tableRow])
    #curdoc().add_root(r)
    return r, predicted

# In[11]:
def analyze(data_p):
    #networks = runNN(df)
    #pickle.dump(networks,open('network.p','wb'))
    (nnCol, predicted) = plots(data_p)
    return nnCol, predicted

