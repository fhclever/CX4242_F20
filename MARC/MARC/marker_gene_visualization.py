def data_processing(df):
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier

    train_size = 10000  # subject to change, I set it to 27848 to run the classifer faster for demo purpose only
    train_data = df[0:train_size]
    train_data = train_data.fillna(0)

    # export the list of cell class
    cell_class = train_data.Cell_class.unique()

    gene_list_array = []
    importance_array = []

    # in this part we will perform OVA Decision Tree to see the most important feature or marker genes
    for i in range(len(cell_class)):
        cell_class_id = i
        cell_data = train_data[:]  # copy dataframe for modification

        # modify data to a binary classification
        cell_data.loc[(df['Cell_class'] == cell_class[cell_class_id]), 'Cell_class'] = "positive"
        cell_data.loc[(df['Cell_class'] != cell_class[cell_class_id]), 'Cell_class'] = "negative"

        # Separate data attributes and label for cell types classification
        y = cell_data["Cell_class"]
        X = cell_data.drop(
            ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Behavior', 'Bregma', 'Centroid_X', 'Centroid_Y', 'Cell_class',
             'Neuron_cluster_ID'], axis=1)

        # Build a forest and fit the data
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        forest.fit(X, y)

        # Store importances in an array
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        indices = indices[0:10]
        counts = []
        for f in range(10):
            counts.append(importances[indices[f]])
        importance_array.append(counts)

        # Store gene list in an array
        genes_list = list(X)
        temp_genes = []
        for f in range(10):
            temp_genes.append(genes_list[indices[f]])
        genes_list = temp_genes
        gene_list_array.append(genes_list)
    return gene_list_array, importance_array, cell_class


def gene_visualization(gene_list_array, importance_array, cell_class, output_file_name):
    from bokeh.io import output_file, show
    from bokeh.layouts import column, row
    from bokeh.models import Div, ColumnDataSource, CustomJS, Dropdown, Select, DataTable, DateFormatter, TableColumn
    from bokeh.palettes import Spectral6
    from bokeh.plotting import figure
    from bokeh.transform import factor_cmap


    # Description
    div1 = Div(text="""We perform the One-vs-All Decision Tree to identify the most important genes for each cell types. To see the detail analysis for each cell types, choose below:
    """,
               width=400, height=60)
    div2 = Div(text="""In the plot, the importance of genes by descending order is visualized. On the x-axis are gene index, and on the y-axis are the gene importances. Hover on the bar to see details.
    """,
               width=400, height=60)
    div3 = Div(text="""
    """,
               width=400, height=120)

    # plot
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
               "#17becf"]

    genes_list = ['gene_0', 'gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5', 'gene_6', 'gene_7', 'gene_8', 'gene_9']
    counts = importance_array[0]
    source = ColumnDataSource(data=dict(genes_list=genes_list, counts=counts, gene_array=gene_list_array[0]))

    TOOLTIPS = [
        ("Gene:", "@gene_array"),
        ("Importances:", "@counts"),
    ]

    p = figure(x_range=genes_list, plot_height=350, tooltips=TOOLTIPS, toolbar_location='right',
               title="Marker genes by importances in determining cell types")
    p.vbar(x='genes_list', top='counts', width=0.9, source=source,
           line_color='white', fill_color=factor_cmap('genes_list', palette=palette, factors=genes_list))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    # selection
    menu = []
    for cell_type in cell_class:
        menu.append(cell_type)

    selecthandler = CustomJS(
        args=dict(source=source, menu=menu, gene_list_array=gene_list_array, importance_array=importance_array), code="""
       var data = source.data;
       console.log('Tap event occurred at x-position: ' + cb_obj.value);

       var index = 0
       for (var i = 0; i < menu.length; i++) {
             if (cb_obj.value==menu[i]) {
                 index = i
             }
       }

       data['counts'] = importance_array[index]
       data['gene_array'] = gene_list_array[index]
       source.change.emit();
    """)

    select = Select(title="Select cell types:", value=menu[0], options=menu)
    select.js_on_change("value", selecthandler)

    # data table report
    columns = [
        TableColumn(field="gene_array", title="Gene"),
        TableColumn(field="counts", title="Importance"),
    ]
    data_table = DataTable(source=source, columns=columns, width=400, height=280, background='white')

    # set visualization layput
    col1 = column(div1, div2, select, data_table)
    col2 = column(div3, p)
    layout = row(col1, col2)

    return layout


def analyze(df):
    (gene_list_array, importance_array, cell_class) = data_processing(df)
    output_file_name = 'marker_genes_demo.html'
    visual_plot = gene_visualization(gene_list_array, importance_array, cell_class, output_file_name)
    return visual_plot

