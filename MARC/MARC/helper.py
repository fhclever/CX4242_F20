import math

def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

# Filter out irrelevant columns
def genes_only(df):
    return df.drop(['Cell_ID','Animal_ID','Animal_sex','Behavior','Bregma','Centroid_X','Centroid_Y',
        'Cell_class', 'Neuron_cluster_ID'], axis = 1)

# Drop null rows
def drop_empty_rows(df):
    return df.dropna(axis = 1)
