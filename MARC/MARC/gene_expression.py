import pandas as pd
import numpy as np
import scipy as sp
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import math

def groupBy_avg_expression(df, column, gene_names):
    start = pd.DataFrame(index = df.index)
    for gene_name in gene_names:
        avgs = df.groupby(column).mean()
        sems = df.groupby(column).sem().rename(columns = {gene_name: gene_name + '_sem'})
        counts = df.groupby(column).count().rename(columns = {gene_name: gene_name + '_count'})
        grouped_df = pd.merge(avgs[gene_name], sems[gene_name + '_sem'], left_index = True,
            right_index = True, how='inner')
        grouped_df = pd.merge(grouped_df, counts[gene_name + '_count'], left_index = True,
            right_index = True, how='inner').sort_values(by = [gene_name], ascending = False)
        if gene_name == gene_names[0]:
            start = grouped_df
        else:
            start = pd.merge(grouped_df, start, left_index = True,
                right_index = True, how='inner')
    return start

def groupBy_avg_expression_correl(df, column, gene_names):
    if type(gene_names) != list or len(gene_names) != 2:
        print("Second argument must be a list of two genes.")
        return None
    # Get the cell types
    cell_types = set(df[column])
    # Set up output
    correls = {}
    for cell_type in cell_types:
        # Filter out the undesired cell types
        grouped = df[df[column] == cell_type]
        # If all gene expression is 0.0
        if (grouped[gene_names[0]] == 0).all() or (grouped[gene_names[1]] == 0).all():
            correls[cell_type] = 0
            continue
        # If there is only one sample
        if (grouped.shape[0] <= 1):
            correls[cell_type] = 0
            continue
        # Get the correlation between the two genes
        corr = grouped[gene_names[0]].corr(grouped[gene_names[1]])
        correls[cell_type] = corr
    out = pd.DataFrame.from_dict(correls, orient = 'index').rename(columns =
        {0: 'correl'})
    sorted_df = out.sort_values(by = 'correl', ascending = False)
    sorted_df[column] = sorted_df.index
    return sorted_df