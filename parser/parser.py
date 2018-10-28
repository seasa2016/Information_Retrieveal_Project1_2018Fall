import xml.etree.ElementTree as ET
import json
import sys

class parser:
    name_parser={
        "SubtaskA_Skip_Because_Same_As_RelQuestion_ID":"SubtaskA_Skip_Because_Same_As_RelQuestion_ID",
        "version":"version",
        "xml":"xml",
        "RELQ_RANKING_ORDER":"ORDER",
        "RelQSubject":"Subject",
        "RELQ_USERID":"USERID",
        "THREAD_SEQUENCE":"THREAD_SEQUENCE",
        "RelCText":"Text",
        "RELC_USERNAME":"USERNAME",
        "ORGQ_ID":"ID",
        "RELQ_USERNAME":"USERNAME",
        "RELC_RELEVANCE2RELQ":"RELEVANCE2RELQ",
        "RELQ_ID":"ID",
        "RELC_DATE":"DATE",
        "RelQBody":"Body",
        "OrgQuestion":"OrgQuestion",
        "RELQ_CATEGORY":"CATEGORY",
        "RELQ_DATE":"DATE",
        "Thread":"Thread",
        "RelQuestion":"RelQuestion",
        "RelComment":"Comment",
        "RELC_RELEVANCE2ORGQ":"RELEVANCE2ORGQ",
        "OrgQBody":"Body",
        "OrgQSubject":"Subject",
        "RELQ_RELEVANCE2ORGQ":"RELEVANCE2ORGQ",
        "RELC_USERID":"USERID",
        "RELC_ID":"ID"
    }
    data = {}

    def __init__(self,file,task='taskA'):
        """
            you can input as a string or a list of string
        """
        if(isinstance(file,list)):
            for f in file:
                tree = ET.parse(f)
                root = tree.getroot()    
                out = self.convert(root)
                self.deal(out,task)
        else:
            tree = ET.parse(file)
            root = tree.getroot()
            out = self.convert(root)[1]
            self.deal(out,task)
        with open('p','w') as f:
            json.dump(self.data,f,indent=4)
        
    def deal(self,out,task='taskA'):
        """
            here we convert the dict to the final data
        """
        #return the question comment pair
        if(task == 'taskA'):
            #here extract from the sub comment
            if('OrgQuestion' in out):
                for _ in out['OrgQuestion']:

                    temp_op = { name:_[name] for name in ["ID","Subject","Body"]}
                    temp_op['Comment'] = []
                    if("SubtaskA_Skip_Because_Same_As_RelQuestion_ID" not in _["Thread"]):
                        temp_rp = {name:_["Thread"][0]["RelQuestion"][name] for name in ["ID","CATEGORY","DATE","USERID","USERNAME","Subject","Body"]}
                        temp_rp['Comment'] = []
                    else:
                        temp_rp = None

                    for tt in _['Thread'][0]['Comment']:
                        temp_op['Comment'].append({name:tt[name] for name in ["ID","DATE","USERID","USERNAME","RELEVANCE2ORGQ","Text"]})
                        if(temp_rp):
                            temp_rp['Comment'].append({name:tt[name] for name in ["ID","DATE","USERID","USERNAME","RELEVANCE2RELQ","Text"]})
                    try:
                        self.data[temp_op['ID']]['Comment'].extend(temp_op['Comment'])
                    except KeyError:
                        self.data[temp_op['ID']]  = temp_op.copy()
                    if(temp_rp is not None):
                        #print(temp_rp)
                        try:
                            self.data[temp_rp['ID']]['Comment'].extend(temp_rp['Comment'])
                        except KeyError:
                            self.data[temp_rp['ID']]  = temp_rp.copy()

            if('Thread' in out):
                for temp in out['Thread']:
                    temp_rp = {name:temp["RelQuestion"][name] for name in ["ID","CATEGORY","DATE","USERID","USERNAME","Subject","Body"]}
                    try:
                        temp_rp['Comment'] = temp["Comment"]
                    except KeyError:
                        continue

                    try:
                        self.data[temp_rp['ID']]['Comment'].extend(temp_rp['Comment'])
                    except KeyError:
                        self.data[temp_rp['ID']]  = temp_rp.copy()
        
        #return the question question pair
        elif(task == 'taskB'):
            if('OrgQuestion' in out):
                for temp in out['OrgQuestion']:
                    temp_op = { name:temp[name] for name in ["ID","Subject","Body"]}
                    temp_op["RelQuestion"] = [temp['Thread'][0]["RelQuestion"]]
                    

                    try:
                        self.data[temp_op["ID"]]["RelQuestion"].extend(temp_op["RelQuestion"])
                    except KeyError:
                        self.data[temp_op["ID"]] = temp_op.copy()

        #return the full data
        elif(task == 'taskC'):
            if('OrgQuestion' in out):
                for temp in out['OrgQuestion']:
                    temp_op = {name:temp[name] for name in ["ID","Subject","Body"]}
                    temp_op["Thread"] = [{name:temp["Thread"][0][name] for name in ["RelQuestion","Comment"]}]
                    
                    try:
                        self.data[temp_op["ID"]]["Thread"].extend(temp_op["Thread"])
                    except KeyError:
                        self.data[temp_op["ID"]] = temp_op.copy()
        else:
            raise ValueError('the input task should be in ("taskA","taskB","taskC")')
    def convert(self,root):
        """
            here we convert the xml file to the dict
        """
        data = dict()
        for name in root.attrib:
            data[self.name_parser[name]] = root.attrib[name]
        
        for child in root:
            if(len(child) == 0):
                data[self.name_parser[child.tag]] = child.text
            else:
                out = self.convert(child)
                if(out[0] == 'Thread' or out[0] == 'OrgQuestion'):
                    try:
                        data[out[0]].append(out[1])
                    except KeyError:
                        data[out[0]] = [out[1]]

                elif(out[0] in data):
                    try:
                        data[out[0]].append(out[1])
                    except AttributeError:
                        data[out[0]] = [data[out[0]]]
                        data[out[0]].append(out[1])
                else:
                    data[out[0]] = out[1]
        return self.name_parser[root.tag],data

    def iterator(self):
        for name in self.data:
            yield(self.data[name])

def main():
    parser(sys.argv[1],sys.argv[2])
if(__name__ == '__main__'):
    main()


