# ML Adaboost

# CSDS 440 Assignment 3
## The repository:
https://csevcs.case.edu/git/2020_fall_440_15
## Project files:
Located under P3 in the overarching git repository.
## Dependencies:
progressbar
argparse
numpy

## Final Commit Progress (11/20/2020)
### Calvin Vuong
    - Implemented code for research extension.
    - General debugging.
    - Notebook:
	- Answer parts (a) and (b) of writeup.
	- Wrote motivation, hypothesis, and results section of research extension.
### Kris Zhao
    - Input arguments:
        - Modified previous input arguments to conform to the new project
    - Progressbar:
        - Added a progressbar to indicate the progress for a run
        - k fold increments one for each iteration of each fold
        - Whole dataset counts iterations
    - Generalization:
        - Cleaned up the code so that the we did not need if statements based on strings.
        - added gen_learn for each learning algorithm.
    - Notebook:
        - Wrote some discussion for exploration.

## First Commit Progress (11/13/2020)
### Calvin Vuong
    - Modify dtree, nbayes, and logreg code to work with weighted examples.
    - Write the cross-validation and "training whole" functions for boost.
    - Debug boosting code: fixes in ensemble voting and classification.
    - Write python script to test boosting algorithm for a range of boosting iterations.
### Kris Zhao
    - Boosting Algorith:
        - Generalalized classify methods for easier classifier calling regardless of nbayes, logreg, or dtree.
        - Implementation of the iterative update of boosting.
        - Implementation of the example weight updates for each iteration.
        - Implementation of the classifier weights
        - Modification to loop over iterations and do the boosted learning.
        - Created the final evaluation of the boosted classifiers (majority vote, for each example).

## Data folder structure:
Data will not be pushed to the git repository. In order to maintain the same data folder structuring, I have created each folder but ignored the .data, .info and .names files.

It will be the responsibility of those running the code to replicate the data. Our data structuring assumptions are as follows:
* spam:
    * spam.data
    * spam.info
    * spam.names
* volcanoes:
    * images/
        * chips.png
        * img1.png  
        ...
        * img9.png
    * volcanoes.data
    * volcanoes.info
    * volcanoes.names
* voting:
    * voting.data
    * voting.info
    * voting.names

## Jupyter Notebook:
We will prepare a Jupyter Notebook for this programming assignment. The directory for the notebook will be in P2 (this directory).

In order to install jupyter, it is suggested to use pip:

```pip install notebook```

Then, once it is complete installing, navigate to the P1 directory. From here, simply run:

```jupyter notebook```

This will open up the notebook server at the default internal ip at the 8888 port. You are then able to open the jupyter_notebook file and view the notebook.