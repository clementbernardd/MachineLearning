# Project using TMDB dataset : Success Movie

This is a project that deals with TMDB dataset from Kaggle.

## Getting Started

You can only download the notebook and run it cell by cell. A python file is available and require to run the notebook.
There is also the located which is placed in the Data directory. 
Then, there is a directory where there are report of our projects (and the explanation of our approach). Be careful because the final report is not up-to-date : the final results are not reliable.  


## Explanation of the project 

### Context
 
It is hard for movie industry to know whether a movie will have success or not.
Nevertheless, it seems that some companies have found a way to attract spectators. But there are still cases where high budget films are flopping and vice versa. 
The public has a lot of different interests, and some films succeeded whereas no one would have predicted that. 
Thus, some actors or companies seem to be enjoyed a lot, more than others. 
In this case, are there rules that can be made to say if a movie will be accepted to its public ?

### Dataset
 
We used a dataset from Kaggle. This dataset comes from TMDb which is one of the most popular source for movie contents. It contains around 5000 different movies and 30 features. These features are divided into 2 csv files. One for the general information about the movie (budget, gender, language, title, …) and the other one for more details (like the casting). We also used a dataset from Kaggle with Oscar results.

### Pre-processing

The pre-processing work is detailed in Results/middle_project.pdf. You can also run the notebook in Model/success_movie.ipynb. 
There is also a way to see the results of the notebook with the html version in Results/Preprocessing.html. 
To summarize quickly the work done, we tried to get the best of our features by creating new features, deleting the useless ones (or the ones that we didn't want to treat because of a lack of time, such as the strings). Then we also used K encoding for few features. 

### Model 

The models used and the meaning of the results are described in a pdf in Results/finalReport.pdf. 
Note that the results described were quite different from those obtained in the notebook. 
To summarize, we used supervised regression and classification with different models. We didn't focus on how to perform the best of each model, but a way to compare and to test each model to have a general idea. In this case, we used a multi layer regressor or classifier but we knew that it wasn't quite relevant because we didn't take time to get the best hyperparameters. 

### Results 

The results are described either on Results/finaleReport.pdf or in the notebook. It seems that the best result was after using all the data (that is to see not spliting the data by genre) with a linear regression. 

## Team

This project was done during 3 months at Eurecom by Maxence Brugères and Clément Bernard (me). 





 






