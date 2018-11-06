--File: XML2df.py--

class: data_preprocessing

Input: 
1. A list of files which are going to be transformed. (file type: .xml)
2. 'task_A', 'task_B' or 'task_C'

* Notice that the fileS created in 2015 are going to be collided with task B and task C. 

Output: 
A pandas dataframe with all attributes the task needed. (pd.DataFrame)


ex.
tr_file_listA = ['data/train/SemEval2016-Task3-CQA-QL-train-part2.xml', 
                 'data/train/SemEval2016-Task3-CQA-QL-train-part1.xml',
                 'data/train/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml',
                 'data/train/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml',
                 'data/train/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml']

train_A = data_preprocessing(tr_file_listA, 'task_A').concat_dfs()
