# Visualization

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl
import pandas_profiling

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import pylab

# Get Missing, Zeros, mean, median, standard deviation, min, 25%, 50%, 75%, max , count, distinct count, cardinality
# Pairwise plot ( scatter, box )
# Get corelation matrix between all the numeric varibles. ( heat map )
# Outlier explorations (box and whisker plots*)

# correlation matrix input dataframe - COMPLETED
def getCorrelationMatrix (df_in):
    
    # get the numeric and string datatypes
    numericList = list(df_in.select_dtypes(include=['int64','float64']).columns)
    stringList = list(df_in.select_dtypes(include=['O']).columns)

    # Get correlation data
    corr = df_in[numericList].corr()
    corr.fillna(0, inplace=True)
    corr.style.background_gradient().set_precision(2)

    f, ax = pl.subplots(figsize=(10, 8))
    corr = df_prop_all_dim_trimmed.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    return

# correlation matrix Dendogram : COMPLETED
def getCorrelationDendogram (df_in):
    
    numericList = list(df_in.select_dtypes(include=['int64','float64']).columns)

    corr = df_in[numericList].corr()
    corr.fillna(0, inplace=True)

    Z = linkage(corr, 'average')

    pl.figure(figsize=(25, 10))
    labelsize=20
    ticksize=15
    pl.title('Hierarchical Clustering Dendrogram ', fontsize=labelsize)
    pl.xlabel('Variables', fontsize=labelsize)
    pl.ylabel('Distance', fontsize=labelsize)
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels = corr.columns
    )
    pylab.yticks(fontsize=ticksize)
    pylab.xticks(rotation=-90, fontsize=ticksize)
    pl.show()
    
    return

# correlation matrix clustered - COMPLETED
def getCorrelationClustered (df_in):
    
    # Two pass clustering
    # 1-We cluster the corr matrix
    #   We sort the data according to this clustering
    # 2-For cluster bigger than a threshold we cluster those sub-clusters
    #   We sort the data according to these clustering
    # Source Link : https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
    # Other link : https://thelonenutblog.wordpress.com/2017/03/30/correlation-matrix-clustering/

    import scipy
    import scipy.cluster.hierarchy as sch
    
    numericList = list(df_in.select_dtypes(include=['int64','float64']).columns)

    cluster_th = 4

    corr = df_in[numericList].corr()
    corr.fillna(0, inplace=True)

    X = corr.values

    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')

    columns = [df_in[numericList].columns.tolist()[i] for i in list(np.argsort(ind))]
    df_in[numericList] = df_in[numericList].reindex(columns, axis=1)

    unique, counts = np.unique(ind, return_counts=True)
    counts = dict(zip(unique, counts))

    i = 0
    j = 0
    columns = []
    for cluster_l1 in set(sorted(ind)):
        j += counts[cluster_l1]
        sub = df_in[numericList][df_in[numericList].columns.values[i:j]]
        if counts[cluster_l1]>cluster_th:
            sub_corr = sub.corr()
            sub_corr.fillna(0, inplace=True)
            X = sub_corr.values
            d = sch.distance.pdist(X)
            L = sch.linkage(d, method='complete')
            ind = sch.fcluster(L, 0.5*d.max(), 'distance')
            col = [sub.columns.tolist()[i] for i in list((np.argsort(ind)))]
            sub = sub.reindex(col, axis=1)
        cols = sub.columns.tolist()
        columns.extend(cols)
        i = j
    df_in[numericList] = df_in[numericList].reindex(columns, axis=1)

    plot_corr(df_in[numericList], 18)
    
    return

# Plot Correlation - COMPLETED
def plot_corr(df,size=10): 
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    %matplotlib inline
    import matplotlib.pyplot as plt

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

# Get Summary Statistics printRes ( 1 / 0 ) whether to print the results - COMPLETED
def getSummaryStats(df_in ):
    
    # get the numeric and string datatypes
    # numericList = list(df_in.select_dtypes(include=['int64','float64']).columns)
    # stringList = list(df_in.select_dtypes(include=['O']).columns)
    # dateList = list(df_in.select_dtypes(include=['datetime']).columns)

    summaryStatDF = df_in.describe(include = 'all').transpose()

    # One option is to just print and other is to send the summary dataframe back
    print (summaryStatDF)
    return

# Get Profile Report - COMPLETED
def getSummaryProfileReport(df_in,outputfileHTML)
    
    # https://github.com/pandas-profiling/pandas-profiling
    
    if outputfile == "":
        return pandas_profiling.ProfileReport(df_in) 
    else:
        profile = pandas_profiling.ProfileReport(df_in)
        profile.to_file(outputfile=outputfileHTML)
        print("Saved Profile reporting ",outputfileHTML)
        return

    
# Get Pairwise plot
def getPairWisePlot(df_in):

    import seaborn as sns
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(df_in)

    return

# Normalization of all the numeric fields
def setNormalization(df_in): - COMPLETED
    
    # get the numeric and string datatypes
    numericList = list(df_in.select_dtypes(include=['int64','float64']).columns)
    stringList = list(df_in.select_dtypes(include=['O']).columns)
    
    # Normalizing all the numeric columns
    cols_to_norm = numericList
    df_in[cols_to_norm] = df_in[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
 
    return df_in