import pandas as pd
import numpy as np
import sys

task = str(sys.argv[1])
data = pd.read_csv('task'+str(task)+'/test.csv', index_col=0, header=0)
preds = pd.read_csv('task'+str(task)+'/'+task+'_test_results.tsv', header=None, delimiter='\t')

def output_result(test_2017, X_test, task):
    df = pd.DataFrame()
    if task=='B':
        df[0] = test_2017['ORGQ_ID']
        df[1] = test_2017['RELQ_ID']
    elif task=='A':
        df[0] = test_2017['RELQ_ID']
        df[1] = test_2017['RELC_ID']
    else:
        df[0] = test_2017['ORGQ_ID']
        df[1] = test_2017['RELC_ID']      
    df[2] = [1] * len(test_2017)
    df[3] = preds[1]
    df[4] = np.where(np.array(preds[1])>0.5, 'true', 'false')
    
    return df
    
output_pred = output_result(data, preds, task)
output_pred.to_csv('predictions_task'+str(task)+'_bert.pred', sep='\t', header=False, index=False)
