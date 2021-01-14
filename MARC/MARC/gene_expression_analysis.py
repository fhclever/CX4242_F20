from bokeh.plotting import figure, output_file, show, save, ColumnDataSource
from bokeh.transform import dodge
import gene_expression
from bokeh.io import curdoc
from bokeh.transform import dodge
from bokeh.layouts import column, layout, row
from bokeh.models import CustomJS, MultiChoice, RadioButtonGroup, Div, Select, FileInput, Panel, Tabs
import pandas as pd

def analyze(df):
    drop_these = ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Bregma', 'Centroid_X', 'Centroid_Y',
                  "Cell_class", "Behavior", "Neuron_cluster_ID"]
    temp = df.copy()
    for i in drop_these:
        if i in df.columns:
            temp = temp.drop(i, axis=1)
    div1 = Div(text="""
        In the section, we summarize the average expression of genes. Select genes and cell type,
        behavior, or inhibitor/excitatory cell cluster.
    """,
               width=400, height=100)
    div2 = Div(text="""
        In the section, we summarize the correlation between a pair of genes by cell type or behavior.
    """,
               width=400, height=100)

    OPTIONS = temp.columns.tolist()
    multi_choice = MultiChoice(title='Genes', value=OPTIONS[0:2], options=OPTIONS, max_items=5,
                               placeholder='Select genes.')
    LABELS = ["Cell_class", "Behavior", "Neuron_cluster_ID"]
    radio_button_group1 = RadioButtonGroup(labels=LABELS, active=0)
    LABELS2 = ["Cell_class", "Behavior", "Neuron_cluster_ID"]
    radio_button_group2 = RadioButtonGroup(labels=["Cell_class", "Behavior"], active=0)
    select1 = Select(title="Gene 1:", value=OPTIONS[0], options=OPTIONS)
    select2 = Select(title="Gene 2:", value=OPTIONS[1], options=OPTIONS)

    p = figure(
        y_axis_label='Average gene expression',
        plot_width=800,
        plot_height=600
    )
    q = figure(
        y_axis_label='Pearson coefficient',
        plot_width=800,
        plot_height=600
    )

    col1 = column(div1, radio_button_group1, multi_choice)
    col2 = column(div2, radio_button_group2, select1, select2)
    l = row(col1, p)
    m = row(col2, q)

    def update_p():
        global p
        temp_genes = multi_choice.value
        drop_these = ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Bregma', 'Centroid_X', 'Centroid_Y',
                      "Cell_class", "Behavior", "Neuron_cluster_ID"]
        drop_these.remove(LABELS[radio_button_group1.active])
        temp = df.drop(drop_these, axis=1).dropna(axis=0)
        src = ColumnDataSource(gene_expression.groupBy_avg_expression(temp, LABELS[radio_button_group1.active],
                                                                      temp_genes))
        if src.data == []:
            return 1
        p = figure(
            title="Average gene expression by cell type: " + str(temp_genes),
            x_axis_label=LABELS[radio_button_group1.active],
            y_axis_label='Average gene expression',
            x_range=src.data[LABELS[radio_button_group1.active]]
        )
        gap = -0.1
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
                   "#17becf"]
        for i, gene in enumerate(temp_genes):
            p.vbar(x=dodge(LABELS[radio_button_group1.active], .2 * i + gap, range=p.x_range),
                   top=gene, width=0.1, source=src,
                   color=palette[i % len(palette)], legend_label=gene)
        p.legend.location = "top_right"
        p.legend.orientation = "horizontal"
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1

        l.children[1] = p
        return 0

    def update_q():
        global q
        temp_genes = [select1.value, select2.value]
        drop_these = ['Cell_ID', 'Animal_ID', 'Animal_sex', 'Bregma', 'Centroid_X', 'Centroid_Y',
                      "Cell_class", "Behavior", "Neuron_cluster_ID"]
        drop_these.remove(LABELS2[radio_button_group2.active])
        temp = df.drop(drop_these, axis=1).dropna(axis=0)
        src = ColumnDataSource(gene_expression.groupBy_avg_expression_correl(temp,
                                                                             LABELS2[radio_button_group2.active],
                                                                             temp_genes))
        if src.data == []:
            return 1
        q = figure(
            title="Average gene expression by cell type: " + str(temp_genes),
            x_axis_label=LABELS2[radio_button_group2.active],
            y_axis_label='Average gene expression',
            x_range=src.data[LABELS2[radio_button_group2.active]]
        )
        q.vbar(x=LABELS2[radio_button_group2.active], top='correl', width=0.2, source=src,
               color='#1f77b4')
        q.x_range.range_padding = 0.1
        q.xaxis.major_label_orientation = 1

        m.children[1] = q
        return 0

    update_p()
    update_q()

    def callback_button1(new):
        update_p()

    def callback_button2(new):
        update_q()

    def callback_choice(attr, new, old):
        update_p()

    def callback_select(attr, new, old):
        update_q()

    radio_button_group1.on_click(callback_button1)
    radio_button_group2.on_click(callback_button2)
    multi_choice.on_change("value", callback_choice)
    select1.on_change('value', callback_select)
    select2.on_change('value', callback_select)
    GEAcol = column(l, m)
    return GEAcol