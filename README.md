
# Kaggle's  Titanic Challenge on Dataiku!



# Introduction
![enter image description here](/images/titanic.jpg)
_Titanic_  was under the command of Capt.  [Edward Smith](https://en.wikipedia.org/wiki/Edward_Smith_(sea_captain) "Edward Smith (sea captain)"), who also  [went down with the ship](https://en.wikipedia.org/wiki/The_captain_goes_down_with_the_ship "The captain goes down with the ship"). The ocean liner carried some of the wealthiest people in the world, as well as hundreds of emigrants from  [Great Britain and Ireland](https://en.wikipedia.org/wiki/United_Kingdom_of_Great_Britain_and_Ireland "United Kingdom of Great Britain and Ireland"),  [Scandinavia](https://en.wikipedia.org/wiki/Scandinavia "Scandinavia")  and elsewhere throughout Europe, who were seeking a new life in the United States. The first-class accommodation was designed to be the pinnacle of comfort and luxury, with a gymnasium, swimming pool, libraries, high-class restaurants and opulent cabins. 

# Installation
On this repository, you may find my personal projects related to Machine Learning, EDA, Python Jupyter Notebook and couple of Visualization based on the Dataiku Platform exported standard files. Most of the datasets I've been working with, downloaded from Kaggle. Installation pretty straight forward. Simply download the whole set as a single project as a ZIP files, everything have been flattened out with plain text files, and no SQL dump was involved, so there wouldn't be any missing system dependencies issue.

# DataFlow
![Titanic Dataflow](/images/dataflow.png)
This is how I created the data flow visualization process, and by the end of it, I'm applying, both the [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression%29.) and the [Decision Tree Model](https://en.wikipedia.org/wiki/Decision_tree_learning) to supply the [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) challenge.

# Features Handling
![Titanic Features Handling](/images/features-handling.png)
Many of the features of the dataset, have been modified through [One Hot Encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) method, that way the ML algorithm would understand them and decipher them better by changing them to categorical units.

## Initial Dataset Features
![Titanic Initial Dataset Handling](/images/initial-dataset.png)

## Data Dictionary
- `Survived`: 0 = No, 1 = Yes
- `pclass`: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
- `sibsp`: # of siblings / spouses aboard the Titanic
- `parch`: # of parents / children aboard the Titanic
- `ticket`: Ticket number
- `cabin`: Cabin number
- `embarked`: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

By default, the initial dataset coming from [Kaggle's](https://www.kaggle.com/c/titanic/data) challenge page would give you the above dataset features at hand. But we'll try to optimize them to something much more Machine Learning friendly looking dataset. And this is how I did it.

### Name Column

This is something you would normally do in Python to extract the Name information from the dataset. The objective is to get the Title information, so that you may utilize them into something more categorical, so that the Machine Learning algorithm could understand them better.
 
    train_test_data = [train, test] # combining train and test dataset
    for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract( '([A-Za-z]+)\.', expand=False)  

Here's the similar method in Dataiku, in a way they produce the similar output, through their Data manipulation recipes canvas.
![Cleaning The Name Column](/images/cleaning-name.png)

#### Name Remapping
Now let's map a categorical number to depict those Title values.
- Mr : 0  
- Miss : 1  
- Mrs: 2  
- Others: 3

You could do the similar in python with these following codes:

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                     "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
    for dataset in train_test_data:
        dataset['Title'] = dataset['Title'].map(title_mapping)

Whereas in the case of Dataiku, it'd be something to this degree.<br /><br />
![Name Mapping](/images/mapping-name.png)

### Sex Column
Now let's move on to the next data column, the sex column. It's pretty common that just about any Gender data values would give you 'Male' or 'Female' attributions. But categorical requirement, would still be needing you to change that to numerical attribution. In python, you could this to have the output:

    sex_mapping = {"male": 0, "female": 1}
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)
Whereas in the prepare recipes, you might do something like this to achieve the similar output. <br /><br />
![Gender Mapping](/images/mapping-gender.png)



# Jupyter Notebooks
- [Jupyter Notebooks](https://github.com/leonism/Dataiku-Titanic/tree/master/ipython_notebooks/.ipynb_checkpoints) 
- [Correlations analysis on Titanic Data with Dataiku](https://github.com/leonism/Dataiku-Titanic/blob/master/ipython_notebooks/.ipynb_checkpoints/Correlations%20analysis%20on%20Titanic_prepared%20%28admin%29-checkpoint.ipynb) 
- [Statistics and tests on multiple populations with Dataiku](https://github.com/leonism/Dataiku-Titanic/blob/master/ipython_notebooks/.ipynb_checkpoints/Statistics%20and%20tests%20on%20multiple%20populations%20on%20train-checkpoint.ipynb "Statistics and tests on multiple populations with Dataiku")
- [Time-Series analytics on Titanic with Dataiku](https://github.com/leonism/Dataiku-Titanic/blob/master/ipython_notebooks/.ipynb_checkpoints/Time-Series%20analytics%20on%20Titanic_prepared%20(admin)-checkpoint.ipynb "Time-Series analytics on Titanic_prepared (admin)-checkpoint.ipynb")


# Disclaimer
And please remember, as this is only a weekend pet project, which I'm doing them for my personal interest only.

