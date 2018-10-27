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
                #self.deal(out,task)
        else:
            tree = ET.parse(file)
            root = tree.getroot()
            out = self.convert(root)[1]
            #self.deal(out,task)
        print(out)
        with open('o','w') as f:
            json.dump(out,f,indent=4)
        
    def deal(self,out,task='taskA'):
        """
            here we convert the dict to the final data
        """
        #return the question comment pair
        if(task == 'taskA'):
            #here extract from the sub comment
            for temp in out['OrgQuestion']:
                for tt in temp['Thread']:
                    pass

            for temp in out['Thread']:
                pass

        #return the question question pair
        elif(task == 'taskB'):
            pass
        #return the full data
        else:
            pass

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
                if(out[0] in data):
                    try:
                        data[out[0]].append(out[1])
                    except AttributeError:
                        data[out[0]] = [data[out[0]]]
                        data[out[0]].append(out[1])
                else:
                    data[out[0]] = out[1]
        return self.name_parser[root.tag],data

def main():
    print("??")
    parser(sys.argv[1])

print("??",__name__)
if(__name__ == '__main__'):
    print("??")
    main()


