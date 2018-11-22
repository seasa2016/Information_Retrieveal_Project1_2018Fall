#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class data_preprocessing:
    
    def __init__(self, file_list, task):
        self.file_list = file_list
        self.task = task
    
    def xml_to_df(self):       
        Task_dfs = []
        
        for file in self.file_list:
            tree = ET.parse(file)
            root = tree.getroot()
            
            
            for question in root:     
                Task_dict = {}

                RELQ_ID = []
                RelQSubject = []
                RelQBody = []
                RELC_IDs = []
                RELC_RELEVANCE2RELQs = []
                RelCTexts = []
                ORGQ_ID = []
                OrgQSubject = []
                OrgQBody = []
                RELQ_RELEVANCE2ORGQ = []   
                # Relevant question ID - A, B
                if question.find('Thread') != None:
                    quest = question.find('Thread') # condition command cause of different types of xml files in 2015 and 2016  
                    RelQuestion = quest.find('RelQuestion')
                    RELQ_ID.append(RelQuestion.attrib['RELQ_ID'])
                    
                    # Relevant question body - A, B
                    RelQSubject.append(RelQuestion.find('RelQSubject').text)
                    RelQBody.append(RelQuestion.find('RelQBody').text)
                    
                    # Relevant comments IDs - A, C       
                    RelComments = quest.findall('RelComment')
                    RELC_IDs.extend([RelComment.attrib['RELC_ID'] for RelComment in RelComments]) # Ten IDs
                    
                    # Relevance between question and comments - A, C    
                    RELC_RELEVANCE2RELQs.extend([RelComment.attrib['RELC_RELEVANCE2RELQ'] for RelComment in RelComments]) # Ten Evaluations
                    
                    # Relevant comments - A, C
                    RelCTexts.extend([RelComment.find('RelCText').text for RelComment in RelComments]) # Ten Comments' texts
                
                if self.task != 'task_A':
                    # Original Question ID - B, C
                    ORGQ_ID.append(question.attrib['ORGQ_ID'])
                    
                    # Original question body - B, C
                    OrgQSubject.append(question.find('OrgQSubject').text)
                    OrgQBody.append(question.find('OrgQBody').text)
                    
                    # Relevance between question and question - B
                    RELQ_RELEVANCE2ORGQ.append(question.find('Thread').find('RelQuestion').attrib['RELQ_RELEVANCE2ORGQ'])
                
                if self.task == 'task_A':
                    Task_dict['RELQ_ID'] = np.repeat(RELQ_ID, int(len(RELC_IDs) / len(RELQ_ID)))
                    Task_dict['RelQSubject'] = np.repeat(RelQSubject, int(len(RELC_IDs) / len(RELQ_ID)))
                    Task_dict['RelQBody'] = np.repeat(RelQBody, int(len(RELC_IDs) / len(RELQ_ID)))
                    Task_dict['RELC_ID'] = RELC_IDs
                    Task_dict['RELC_RELEVANCE2RELQ'] = RELC_RELEVANCE2RELQs
                    Task_dict['RelCText'] = RelCTexts
                    
                    Task_df = pd.DataFrame.from_dict(Task_dict)
                    Task_dfs.append(Task_df)
                
                elif self.task == 'task_B':
                    Task_dict['ORGQ_ID'] = ORGQ_ID
                    Task_dict['OrgQSubject'] = OrgQSubject
                    Task_dict['OrgQBody'] = OrgQBody
                    Task_dict['RELQ_ID'] = RELQ_ID
                    Task_dict['RelQSubject'] = RelQSubject
                    Task_dict['RelQBody'] = RelQBody
                    Task_dict['RELQ_RELEVANCE2ORGQ'] = RELQ_RELEVANCE2ORGQ
                    
                    Task_df = pd.DataFrame.from_dict(Task_dict)
                    Task_dfs.append(Task_df)
                    
                else:
                    Task_dict['ORGQ_ID'] = np.repeat(ORGQ_ID, int(len(RELC_IDs) / len(ORGQ_ID)))
                    Task_dict['OrgQSubject'] = np.repeat(OrgQSubject, int(len(RELC_IDs) / len(ORGQ_ID)))
                    Task_dict['OrgQBody'] = np.repeat(OrgQBody, int(len(RELC_IDs) / len(ORGQ_ID)))
                    Task_dict['RELC_ID'] = RELC_IDs
                    Task_dict['RELC_RELEVANCE2RELQ'] = RELC_RELEVANCE2RELQs
                    Task_dict['RelCText'] = RelCTexts
                    
                    Task_df = pd.DataFrame.from_dict(Task_dict)
                    Task_dfs.append(Task_df)
            
        return Task_dfs
    
    
    def concat_dfs(self):
        Task_dfs = self.xml_to_df()
        Task_data = pd.DataFrame.copy(Task_dfs[0])
        if len(Task_dfs) > 1:
            for df in Task_dfs[1:]:
                df.index = np.arange(Task_data.shape[0], Task_data.shape[0] + len(df))
                Task_data = pd.concat([Task_data, df], 0)
        return Task_data
