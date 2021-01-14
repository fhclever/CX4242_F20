import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import time
from bokeh.plotting import figure, output_file
from bokeh.models.sources import ColumnDataSource
from bokeh.models import HoverTool, Label, LabelSet, Div, Panel, Tabs, CustomJS, RadioButtonGroup
from bokeh.plotting import output_notebook, show
from bokeh.io import output_file, curdoc
from bokeh.io import show
from bokeh.models import CustomJS, RadioButtonGroup
from bokeh.layouts import column, layout, row

def analyze(df):

    def hierarchClustering(df):
        train_size = 1  # subject to change, I set it to 27848 to run the classifer faster for demo purpose only
        df.dropna()
        train_data = df.sample(frac = train_size)
        train_data = train_data.fillna(0)
        y = train_data[["Cell_class", "Behavior"]]
        X = train_data.drop(
            ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Behavior', 'Bregma', 'Centroid_X', 'Centroid_Y', 'Cell_class',
             'Neuron_cluster_ID'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        linkMethods = ['complete', 'average', 'single']
        n_clusters = [2, 3, 4, 5, 6, 7, 8]

        models = []
        si = []
        ar = []
        fm = []
        ch = []
        tics = []
        labels = []

        tic1 = time.perf_counter()
        for cluster in n_clusters:
            tempModels = []
            tempSi = []
            tempCh = []
            tempLabels = []

            tica = time.perf_counter()
            for method in linkMethods:
                model = AgglomerativeClustering(n_clusters=cluster, linkage=method)
                fitted_model = model.fit(X_train)
                tempModels = tempModels + [fitted_model]
                tempSi = tempSi + [metrics.silhouette_score(X_train, model.labels_)]
                tempCh = tempCh + [metrics.calinski_harabasz_score(X_train, model.labels_)]

            models = models + [tempModels]
            si = si + [tempSi]
            ch = ch + [tempCh]
            toca = time.perf_counter()
            tics = tics + [toca - tica]
            toc1 = time.perf_counter()

        numClusters = len(n_clusters)
        numMethods = len(linkMethods)

        (si_Cluster, si_Method, si_Val, si_cIdx, si_mIdx) = (2, 'complete', si[0][0], 0, 0)
        (ch_Cluster, ch_Method, ch_Val, ch_cIdx, ch_mIdx) = (2, 'complete', ch[0][0], 0, 0)

        c = 0
        for cluster in n_clusters:

            m = 0
            for method in linkMethods:
                if si[c][m] > si_Val:
                    (si_Cluster, si_Method, si_Val, si_cIdx, si_mIdx) = (cluster, method, si[c][m], c, m)

                if ch[c][m] > ch_Val:
                    (ch_Cluster, ch_Method, ch_Val, ch_cIdx, ch_mIdx) = (cluster, method, ch[c][m], c, m)

                m = m + 1

            c = c + 1

        methods = [ch_Method, si_Method]
        clusters = [ch_Cluster, si_Cluster]

        Zs = []
        results_array = []

        for i in range(len(methods)):
            Z = linkage(X_train, methods[i])
            Zs = Zs + [Z]
            results = dendrogram(Z, truncate_mode='level', p=clusters[i], no_plot=True)
            results_array = results_array + [results]

        return X_train, y_train, linkMethods, n_clusters, models, clusters, si, ch, Zs, results_array, si_Cluster, si_Method, si_Val, si_cIdx, si_mIdx, ch_Cluster, ch_Method, ch_Val, ch_cIdx, ch_mIdx

    def combineStrings(str1, str2):
        words2 = str2.split(', ')
        for word in words2:
            if word not in str1:
                str1 = str1 + ', ' + word

        return str1

    def makeDendrogram(results, X_train, y_train, modIdx, evalMetrics, methodIdx):
        linkMethods = ['complete', 'average', 'single']
        icoord, dcoord = results['icoord'], results['dcoord']
        labels = list(map(int, results['leaves']))
        ivls = results['ivl']
        X_rel = X_train.reindex(index=labels)
        std = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
        X_scaled = std * (1.0 - 0.0) + 0.0
        y_rel = y_train.reindex(index=labels)
        y_rel.head()

        cell_class = []
        behavior = []
        xs = []
        ys = []
        alpha = []

        for i, tm in enumerate(X_rel.index):
            cell_class = cell_class + [y_rel['Cell_class'].loc[tm]] * len(X_rel)
            behavior = behavior + [y_rel['Behavior'].loc[tm]] * len(X_rel)
            xs = xs + list(np.arange(0.5, len(X_rel) + 0.5))
            ys = ys + [i + 0.5] * len(X_rel)
            alpha = alpha + [0.7]

        alpha = np.zeros(len(xs)).tolist()
        alpha = [alphat + 0.75 for alphat in alpha]
        colors = ["#EEEEEE" for t in alpha]

        data = pd.DataFrame(
            dict(xs=xs, ys=ys, alpha=alpha, colors=colors, behavior=behavior, cell_class=cell_class))

        icoord = pd.DataFrame(icoord)
        icoord = icoord * (data['ys'].max() / icoord.max().max())
        icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (data['xs'].max() / dcoord.max().max())
        dcoord = dcoord.values

        hover = HoverTool()

        hover.tooltips = [
            ("Cell Types", "@cell_class"),
            ("Behaviors", "@behavior"),
            ("Cluster Population", "@numPoints")
        ]


        x_min = 0
        for i, d in zip(icoord, dcoord):
            d = list(map(lambda x: -x, d))

            if min(d) < x_min:
                x_min = min(d)

        global hm
        hm = figure(x_range=[x_min, 20],
                    height=600,
                    width=700
                    )

        tempi = icoord
        tempd = dcoord
        order = []

        while len(order) < len(tempi):
            toRemove =[]
            idx = 0
            for i, d in zip(tempi, tempd):
                if idx not in order:
                    d = list(map(lambda x: -x, d))
                    leafBool = [True for val in d if val == 0]
                    if len(leafBool) == 2:
                        toRemove = toRemove + [idx]
                    elif len(leafBool) == 1:
                        leafIndices = [f for f, z in enumerate([True if val == 0 else False for val in d]) if z][0]
                        if leafIndices == 0:
                            leafIndices = 2
                        else:
                            leafIndices = 1
                        hasInfo = False
                        if len(order) > 0:
                            for o in order:
                                ti = sum(icoord[o])/len(icoord[o])
                                if ti == i[leafIndices]:
                                    hasInfo = True

                        if hasInfo:
                            toRemove = toRemove + [idx]

                    else:
                        leafIndices1 = 1
                        leafIndices2 = 3
                        hasInfo1 = False
                        for o in order:
                            ti = sum(icoord[o])/len(icoord[o])
                            if ti == i[leafIndices1]:
                                hasInfo1 = True
                        hasInfo2 = False
                        for o in order:
                            ti = sum(icoord[o])/len(icoord[o])
                            if ti == i[leafIndices2]:
                                hasInfo2 = True
                        if hasInfo1 and hasInfo2:
                            toRemove = toRemove + [idx]

                idx = idx + 1

            order = order + toRemove

        icoord = icoord[order]
        dcoord = dcoord[order]
        labels = np.array(labels)[order]
        ivls = np.array(ivls)[order]

        idx = 0
        leaves = []
        ds = []
        i_s = []
        splits = []
        split_behaviors = []
        split_classes = []
        split_populations = []

        leaf_is = []
        leaf_ds = []
        leaf_behaviors = []
        leaf_classes = []
        leaf_populations = []

        for i, d in zip(icoord, dcoord):
            d = list(map(lambda x: -x, d))
            ds = ds + [d]
            i_s = i_s + [i]

            leafBool = [True for val in d if val == 0]
            if len(leafBool) == 2:
                cell_classes = [y_rel['Cell_class'].loc[labels[idx]]] * 2 + [
                    y_rel['Cell_class'].loc[labels[idx + 1]]] * 2
                behaviors = [y_rel['Behavior'].loc[labels[idx]]] * 2 + [y_rel['Behavior'].loc[labels[idx + 1]]] * 2
                if '(' not in ivls[idx]:
                    bot = 1
                else:
                    bot = int(ivls[idx][1:-1])
                if '(' not in ivls[idx + 1]:
                    top = 1
                else:
                    top = int(ivls[idx + 1][1:-1])
                numPoints = [bot, bot, top, top]
                splits = splits + [(sum(i)/len(i),d[2])]
                split_behaviors = split_behaviors + [combineStrings(str(behaviors[0]),str(behaviors[3]))]
                split_classes = split_classes + [combineStrings(str(cell_classes[0]), str(cell_classes[3]))]
                split_populations = split_populations+ [2]

                leaf_is = leaf_is + [i[0]] + [i[2]]
                leaf_ds = leaf_ds + [d[0]] + [d[3]]
                leaf_behaviors = leaf_behaviors + [behaviors[0]] + [behaviors[3]]
                leaf_classes = leaf_classes + [cell_classes[0]] + [cell_classes[3]]
                leaf_populations = leaf_populations + [bot] + [top]

            elif len(leafBool) == 1:
                leafIndices = [f for f, z in enumerate([True if val == 0 else False for val in d]) if z][0]
                if leafIndices == 0:
                    nonLeafIndex = 3
                else:
                    nonLeafIndex = 0

                leafBehaviors = [y_rel['Behavior'].loc[labels[idx]]]*2
                leafCellClasses = [y_rel['Cell_class'].loc[labels[idx]]] * 2
                if '(' not in ivls[idx]:
                    bot = 1
                else:
                    bot = int(ivls[idx][1:-1])
                leafPoints = [bot, bot]

                nonLeafTuple = (i[nonLeafIndex], d[nonLeafIndex])
                valIdx = splits.index(nonLeafTuple)
                nonLeafBehaviors = split_behaviors[valIdx]
                nonLeafCellClasses = split_classes[valIdx]
                nonLeafPoints = split_populations[valIdx]

                splits = splits + [(sum(i)/len(i),d[2])]
                split_behaviors = split_behaviors + [combineStrings(str(leafBehaviors[0]), str(nonLeafBehaviors))]
                split_classes = split_classes + [combineStrings(str(nonLeafCellClasses), str(leafCellClasses[0]))]
                split_populations = split_populations + [nonLeafPoints+1]

                if leafIndices == 0:
                    cell_classes = leafCellClasses + [nonLeafCellClasses]*2
                    behaviors = leafBehaviors + [nonLeafBehaviors]*2
                    numPoints = leafPoints + [nonLeafPoints]*2
                else:
                    cell_classes = [nonLeafCellClasses]*2 + leafCellClasses
                    behaviors = [nonLeafBehaviors]*2 + leafBehaviors
                    numPoints = [nonLeafPoints]*2 + leafPoints

                leaf_is = leaf_is + [i[leafIndices]]
                leaf_ds = leaf_ds + [d[leafIndices]]
                leaf_behaviors = leaf_behaviors + [behaviors[leafIndices]]
                leaf_classes = leaf_classes + [cell_classes[leafIndices]]
                leaf_populations = leaf_populations + [bot]
            else:
                nonLeafTuple1 = (i[0], d[0])
                valIdx1 = splits.index(nonLeafTuple1)
                nonLeafBehaviors1 = split_behaviors[valIdx1]
                nonLeafCellClasses1 = split_classes[valIdx1]
                nonLeafPoints1 = split_populations[valIdx1]

                nonLeafTuple2 = (i[3], d[3])
                valIdx2 = splits.index(nonLeafTuple2)
                nonLeafBehaviors2 = split_behaviors[valIdx2]
                nonLeafCellClasses2 = split_classes[valIdx2]
                nonLeafPoints2 = split_populations[valIdx2]

                cell_classes = [nonLeafCellClasses1]*2 + [nonLeafCellClasses2]*2
                behaviors = [nonLeafBehaviors1]*2 + [nonLeafBehaviors2]*2
                numPoints = [nonLeafPoints1, nonLeafPoints1, nonLeafPoints2, nonLeafPoints2]

                splits = splits + [(sum(i) / len(i), d[2])]
                split_behaviors = split_behaviors + [combineStrings(nonLeafBehaviors1, nonLeafBehaviors2)]
                split_classes = split_classes + [combineStrings(nonLeafCellClasses1, nonLeafCellClasses2)]
                split_populations = split_populations + [nonLeafPoints1 + nonLeafPoints2]

            dta = pd.DataFrame(dict(x=d, y=i, cell_class=cell_classes, behavior=behaviors, numPoints=numPoints))

            hm.line(x='x', y='y', line_color='black', source=dta)

            leaf = labels[idx]
            leaves = leaves + [leaf]
            idx = idx + 1

        leaf_text = []
        leaf_size = []
        for idx in range(len(leaf_populations)):
            leaf_text = leaf_text + ['Behavior: ' + str(leaf_behaviors[idx]) + ', Cell Class: '
                                     + str(leaf_classes[idx]) + ', Populations: ' + str(leaf_populations[idx])]
            leaf_size = leaf_size + ["25px"]

        labelSource = ColumnDataSource(data=dict(x=leaf_ds,
                                                 y=leaf_is,
                                                 text=leaf_text,
                                                 size=leaf_size))
        leaf_labels = LabelSet(x='x',
                               y='y',
                               text='text',
                               text_font_size='10px',
                               source=labelSource,
                               x_offset=2,
                               y_offset = -5)

        eval_metrics = ['Calinski Harabasz', 'Silhouette']

        hm.add_layout(leaf_labels)
        hm.add_tools(hover)
        hm.title.text = "Hierarchical Clustering - " + eval_metrics[modIdx] + ' Score: ' + str(float(f"{evalMetrics[modIdx]:.2f}")) + " by " + methodIdx[modIdx] + " linkage method"
        hm.axis.major_tick_line_color = None
        hm.axis.minor_tick_line_color = None
        hm.axis.major_label_text_color = None
        hm.axis.major_label_text_font_size = '0pt'
        hm.axis.axis_line_color = None
        hm.grid.grid_line_color = None
        hm.outline_line_color = None
        return hm

    def update():
        global col1
        modIdx = buttons.active
        (X_train, y_train, linkMethods, n_clusters, models, clusters, si, ch, Zs, results_array, si_Cluster,
         si_Method, si_Val, si_cIdx, si_mIdx, ch_Cluster, ch_Method, ch_Val, ch_cIdx, ch_mIdx) = outTuple

        results = results_array[modIdx]
        hm = makeDendrogram(results, X_train, y_train, modIdx, [ch_Val, si_Val], [ch_Method, si_Method])

        col1.children[2] = hm

    def callback_button(new):
        update()

    outTuple = hierarchClustering(df)
    hm = figure(x_range=[-15, 0],
                height=600,
                width=700
                )

    buttons = RadioButtonGroup(labels=['Calinski Harabasz', 'Silhouette'], active=0)
    modIdx = buttons.active
    (X_train, y_train, linkMethods, n_clusters, models, clusters, si, ch, Zs, results_array, si_Cluster,
     si_Method, si_Val, si_cIdx, si_mIdx, ch_Cluster, ch_Method, ch_Val, ch_cIdx, ch_mIdx) = outTuple
    results = results_array[modIdx]
    hm = makeDendrogram(results, X_train, y_train, modIdx, [ch_Val, si_Val], [ch_Method, si_Method])

    txt = Div(text = "Here, we utilize agglomerative hierarchical clustering to group data by gene expression. Multiple linkage methods are tested and evaluated based on the two metrics below. Select the metric you wish to cluster by.", width = 700, height = 60)
    global col1
    col1 = column(txt, buttons, hm)

    buttons.on_click(callback_button)


    return col1