### h3<data usage>
In the file parser.py, it provides a class which is able to deal with the data we need in this project.

please see the file code_test.py for the example of the class

following will be two example that demonstrate the format that used to save the data.
first for the question pair.

- self.Question `dict`
    - <font color="blue">body </font>>>>>`STRING` : content for the question
    - <font color="blue">subject </font>>>>`STRING` : title for the question
    - <font color="blue">category</font>>>`STRING` : category of the question
    - <font color="blue">userID </font>>>>`STRING` : user id for the question
    - <font color="blue">username </font>>`STRING` : user name for the question
    - <font color="blue">comment</font>>>`LIST` : ten comments for the question
      - <font color="blue">date </font>>>>>`datetime` : ten comments for the question
      - <font color="blue">text </font>>>>>>`STRING` : text for the comment
      - <font color="blue">userID </font>>>>`STRING` : user id for the comment
      - <font color="blue">username </font>>`STRING` : user name for the comment
      - <font color="blue">relevance </font>>`STRING` : [PotentiallyUseful,Good,Bad]

- self.OriQuestion `dict`
  - <font color="blue">body </font>>>>>`STRING` : content for the question
  - <font color="blue">subject </font>>>>`STRING` : title for the question
  - <font color="blue">RelQuestion</font>>>`LIST` : ten relevance for the question
    - <font color="blue">body </font>>>>>`STRING` : content for the question
    - <font color="blue">subject </font>>>>`STRING` : title for the question
    - <font color="blue">category</font>>>`STRING` : category of the question
    - <font color="blue">ORDER </font>>>>`INT` : order of the relevance to the origin question
    - <font color="blue">rel_ori </font>>>>`STRING` : [PerfectMatch,Relevant,Irrelevant]
    - <font color="blue">comment</font>>>`LIST` : ten comments for the question
      - <font color="blue">date </font>>>>>`datetime` : ten comments for the question
      - <font color="blue">text </font>>>>>>`STRING` : text for the comment
      - <font color="blue">userID </font>>>>`STRING` : user id for the comment
      - <font color="blue">username </font>>`STRING` : user name for the comment
      - <font color="blue">rel_q </font>>>>>`STRING` : [PotentiallyUseful,Good,Bad]
      - <font color="blue">rel_ori </font>>>>`STRING` : [PotentiallyUseful,Good,Bad]