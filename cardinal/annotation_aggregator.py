import numpy as np
from snorkel.labeling import LabelModel


ANNOTATIONS_FIXED = ['comment', 'session', 'annotator', 'class']
ANNOTATIONS_CLASS = 'class'

def classmax_random_ties(series):
    unique, counts = np.unique(series, return_counts=True)
    max_classes = unique[counts == counts.max()]
    return np.random.choice(max_classes)
    
def vote_selection(df, class_column=ANNOTATIONS_CLASS, fixed_columns=ANNOTATIONS_FIXED):
    columns = df.columns.values
    columns = np.extract(np.logical_not(np.isin(columns, fixed_columns)), columns).tolist()
    
    if len(columns) == 1:
        columns = columns[0]

    df = df.groupby(columns).agg({ANNOTATIONS_CLASS: classmax_random_ties}).reset_index()
    df['comment'] = ''
    df['session'] = '0'
    df['annotator'] = 'vote'
    
    return df

def snorkel(df, class_column=ANNOTATIONS_CLASS, fixed_columns=ANNOTATIONS_FIXED):
    n_classes = len(df[ANNOTATIONS_CLASS].unique())
    
    # Our goal here is to obtain a dataframe consisting of all annotations of annotators    
    
    columns = df.columns.values
    columns = np.extract(np.logical_not(np.isin(columns, fixed_columns)), columns).tolist()
    
    df = df.groupby(columns + ['annotator'])[ANNOTATIONS_CLASS].first().unstack().fillna(-1)
    
    label_model = LabelModel(cardinality=n_classes, verbose=True)
    label_model.fit(df.values)
    pred = label_model.predict(df.values)
    
    # We take the index as a plain dataframe
    df = df.index.to_frame()

    df['comment'] = ''
    df['session'] = '0'
    df[ANNOTATIONS_CLASS] = pred
    df['annotator'] = 'snorkel'
    
    return df
