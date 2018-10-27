In the file parser.py, it provides a class which is able to deal with the data we need in this project.

please see the file code_test.py for the example of the class

following will be two example that demonstrate the format that used to save the data.
first for the question pair.

- self.Question
    - <font color="blue">id </font>>>>>>>`INT` :question id 
    - <font color="blue">body </font>>>>>`STRING` : content for the question
    - <font color="blue">subject </font>>>`STRING` : title for the question
    - <font color="blue">comment </font>>`LIST` : ten comments for the question
      - <font color="blue">date </font>>>>>>>`datetime` : ten comments for the question
      - <font color="blue">text </font>>>>>>>`STRING` : text for the comment
      - <font color="blue">userID </font>>>>>`STRING` : user id for the comment
      - <font color="blue">username </font>>`STRING` : user name for the comment
      - <font color="blue">relevance </font>>`STRING` : [PotentiallyUseful,Good,Bad]



