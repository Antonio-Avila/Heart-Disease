# Machine Learning with a Heart

The competition was hosted by [**DrivenData**](https://www.drivendata.org/), an data science company that hosts competitions 
similar to Kaggle. 
Typically, DrivenData's competitions focus on some of the challenges affecting society in an attempt to find solutions. 


## Problem Description

A full description of the project can be found via this [link](https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/page/107/).

The goal was to predict the probability of a patient developing heart disease as accurately as possible. A solution to the
problem would possibly allow for heart disease to be detected early or ideally not needing to completely rely on having
a doctor present to make the diagnosis.

The data used for the competition was provided by the [Cleveland Heart Disease Database](http://archive.ics.uci.edu/ml/datasets/statlog+(heart))
via the UCI Machine Learning repository. 
It provided 13 features, most of which were medical readings, from 180 patients ranging from the age, sex, thalium stress test, 
maximum heart rate acheived, and several others. They were meant to be used to predict on a binary class feature, heart_disease_present,
which represented whether or not a patient had heart disease. 


The main challenge resided in how the competition was being scored/evaluated. As mentioned, the data provided a label for whether or not
a patient had heart disease, but the ultimate goal was to predict the probability of a patient having heart disease. Though the probability
could be predicted given the labels, we had to minimize the log loss function by minimizing the unknown probability values. 


## Approach

After some initial research into what the data meant, I dove in and explored the data to determine which features seemed to affect the presence 
of heart disease. Out of the 13 features, 4 were found to be nonsignificant: resting blood pressure, fasting blood sugar, resting ekg results, and serum cholesterol.
There were very small to no differences between the people with heart disease and the people without for the 4 features. Age of the patients was a noteworthy feature.
I expected the people with heart disease to skew towards the older population but that was not the case.  The groups' ages had similar averages but their distributions
were slightly different. There was less variability in the people with heart disease; upon further investigation, I found that this was a result of older males developing
heart disease which made sense since the majority of people with heart disease were males. Thought statistical tests and the models determined that age was insignificant,
I opted to keep it in since it in reality, age should play a factor in developing heart disease.


After the initial data exploration, I went into trying to model the data. First I split the data into training and testing sets using stratified sampling so no features 
were underrepresented in the sets. 

I focused on optimizing a Logistic Regression model since in the medical field interpretability is king, but I also tried a few other models such as XGBoost and Random Forest.
With some fine tuning, the logistic model with regularization performed the best for me. Though I wished I could have looked at the statistic of the model, it was not possible due to the
Sci-kit Learn's and Statsmodel's limitations for further tuning. 

