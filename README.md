# Machine Learning with a Heart

The competition was hosted by [**DrivenData**](https://www.drivendata.org/), an data science company that hosts competitions 
similar to Kaggle. 
Typically, DrivenData's competitions focus on some of the challenges affecting society in an attempt to find solutions. 


I ended up placing 95 out of 3905 competitors, putting me in the top 97.6 percentile, with a score of 0.28975. Not too shaby but could have 
been better.


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

Submissions of the predicitons were limited to 3 a day. They could be used to give an indication of how well the model was actually performing on the "new" data.
The results posted during the competition were based on a portion of the new data we were meant to predict referred to as the Public Score.
At the deadline, all submissions would be judged on the entire, new dataset and would result in a private and final score. 

## Approach

After some initial research into what the data meant, I dove in and explored the data to determine which features seemed to affect the presence 
of heart disease. Out of the 13 features, 4 were found to be nonsignificant: resting blood pressure, fasting blood sugar, resting ekg results, and serum cholesterol.
There were very small to no differences between the people with heart disease and the people without for the 4 features. Age of the patients was a noteworthy feature.
I expected the people with heart disease to skew towards the older population but that was not the case.  The groups' ages had similar averages but their distributions
were slightly different. There was less variability in the people with heart disease; upon further investigation, I found that this was a result of older males developing
heart disease which made sense since the majority of people with heart disease were males. Though statistical tests and the models determined that age was insignificant,
I opted to keep it in since it in reality, age should play a factor in developing heart disease.


After the initial data exploration, I went into trying to model the data. First I split the data into training and testing sets using stratified sampling so no features 
were underrepresented in the sets. 

I focused on optimizing a **Logistic Regression** model since in the medical field interpretability is king, but I also tried a few other models such as *XGBoost* and *Random Forest*.
With some fine tuning, the logistic model with regularization performed the best for me. Though I wished I could have looked at the statistic of the model, it was not possible due to the
Sci-kit Learn's and Statsmodel's limitations for further tuning. 

My best performing model was a Logistic Regression model regularized using an elastic net with an l1 ratio of 0.1 and a regularization penalty of 1.5, trained only on the training set. 
Oddly enough, the same model trained on the full data set performed worse on the new data than the model with a subset of the data. This was odd since more data usually increases model
performance, but perhaps there was something in the testing set that caused the decrease in performance.

## Conclusion and Lessons Learned

- Ultimately, I **placed 95 out of 3905** landing me in the **top 4%** of competitors. 
	- Not bad but could have been better but the grind for shaving off decimal points in error is tough. 
- Placement was suprising given the initial public score. My best model performed better once graded on the entire data instead of just a portion
	- model generalized relatively well to new data, which is usually the goal 
	- never want a model that performs well on the data used to train the model but then performs poorly when introduce to new data
- The GridSearchCV function which finds the best parameters for the model has a log loss function that could had helped find a better model given the evaluation metric
- Done further hypertuning, or instead of using a grid search approach tried a random search or gradient based optimization approach. 
- Tried a few other models such as Probit, similar to Logit function but uses a different link function
	- on the downside, it would be less interpritable since it does not give the log odds as Logistic Regression does
- Performed some error analysis to see where the model broke down or see if there were any discernible patterns in the errors made
- Perhaps would have used R instead of Python since it is easier to look at the model's statistics, ie, BIC score, coefficient p-values, etc (would have been nice to have)
	- couldn't do in Python since the library statsmodel's Logit model is the only Logistic Regression model which provides such values, but it is not very flexible
	- it only performs l1 regularization (LASSO) and since I was using elastic net regularization, I could not use the library to create a similar model as Sci-kit Learn's Logit model.
	- no need to worry about this in R (or at least I am better able to maneuver such problems)
- Curiously, the same model trained with more data performed worse. One would think more data would improve model performance by reducing variation, but it was not the case for me.
