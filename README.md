# WineQualityUsingML: Predicting the Wine Quality using Physicochemical Properties

Wine is a beloved beverage all across the world with a long history. According to the Wine
Institute, the California wine and winegrape sector and allied businesses delivered a total
economic contribution of $57.6 billion annually to the state’s economy and $114.1 billion
annually to the U.S. economy. With a rapidly growing number of wine companies and wine
lovers, there’s been a strong interest in selecting the right kind of wine for the individual. As an
avid wine critic, I was very interested in working with a wine quality dataset from a public
repository, and I wanted to create a machine learning model that would predict a quality of a
wine. The question for this paper is “can we predict the quality of wine using physicochemical
properties?”.


I wanted to explore the relationship between the physicochemical properties of wine and the
quality of the wine. Fortunately, there is a dataset from University of California Irvine that
provides two (red and white) wine quality dataset composed of 12 features. Eleven of the
features provided from this dataset are physicochemical properties of wine (fixed acidity, volatile
acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH,
sulphates, alcohol). For each sample, there is one output which is the grade of the wine
criticized by at least 3 wine experts. The grade ranges from 0 to 10.

After a quick exploratory data analysis, I discovered a challenge for this project.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img1.png)

If can see the third row and second column of both figures above, there is a strong imbalance
and lack of distribution in the quality of the wine. Many of the instances of red or white wine
received the score of 5 or 6. It is expected to have so much 5 or 6 for the scores because
average/mid-level grade are more prevalent as they are affordable and easy to produce. This
raises a question because highly imbalanced data do not perform a great job predicting all class
or value.
Without any data preprocessing, I wanted to test how well our built-in regression models from
Scikit-Learn would be able to predict the wine quality. I performed some tests using several
regression models I learned in class (linear regression, ridge regression, kernel regression,
source vector regression, lasso regression, and elastic regression). Using all 11 features as an
input, I received the following results (no feature selections).

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img2.png)

As you can see, data imbalance is a huge problem. The best accuracy score on both red and
white wine don’t even yield anything above 40%. The result you see above does not take in
consideration of feature selection. Therefore, in my feature selection step, I used three different
methods of feature selection (Pearson’s Correlation, SelectKBest, Feature Importance). My
approach is the choose three features that appear the most from these applications of feature
selection. Feature selection is used in machine learning because you may inadvertently
introduce bias in your model that can result in overfitting.
Pearson’s correlation matrix show us the measure of strength of a linear association between
two variables, negative 1 displays a strong negative correlation and positive 1 displays a strong
positive correlation.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img3.png)

I elected alcohol, volatile acid, and citric acid for red wine later. I will compile the top three
features that will be used for training and testing.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img4.png)

Here in white wine, I chose alcohol, density, chlorides, and volatile acid.
Next, I performed a feature selection using SelectKBest. SelectKBest is use to select features
according to the k highest scores.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img5.png)

From this feature selection, three best attributes are total sulfur dioxide, free sulfur dioxide, and
alcohol. It was interesting to see that total sulfur dioxide had the k highest scores out of all other
attributes.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img6.png)

Select K Best for white wine gives us a similar result; total sulfur dioxide has the highest k score.
Lastly, I performed feature selection using feature importance. Feature importance is done with
extra trees classifier.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img7.png)

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img8.png)

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img9.png)

I will use these selected variables to train and predict wine quality.
What creates this dataset very challenging to tackle is that the quality of the wine is completely
subjective. After a close looking at our regression models’ performance (30% accuracy), I
learned to migrate from regression task to classification task, by converting the continuous
variables of wine quality to something categorical. I separated the quality into three groups respectively (low, mid, high). A low quality wine ranges from grade 0 through 4. A mid quality
wine ranges from grade 5 through 6. And lastly, high quality wine ranges from grade 7 through
10. To achieve this, I used a condition to check if a sample wine falls in any of these ranges I
mentioned above, and I simply labeled 1 for low, 2 for mid, and 3 for high. This way would
greatly reduce the error of prediction, and it gives the model a window to aim.
I used three supervised learning classifiers: source vector machine, random forest classifier,
and neural network. These classifiers were all covered in class.
I began separating the training and testing samples using train_test_split module in scikit learn. I
wanted to weigh my training sample more than the testing samples, so my test size is 30%.
Using the three features I chose from feature selections, I obtained the following results for
Random Forest Classifier, Source Vector Machine, and Neural Network.

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img10.png)

![](https://github.com/heeseungp/WineQualityUsingML/blob/master/img/img11.png)

In conclusion, breaking the continuous variables into categorical variables highly increased the
accuracy results from 30% to 85%. It is very effective and convenient because for any wine
drinker, it is simpler and more intuitive to understand grading using categorical data like low,
mid, high and we are not losing much information when doing this reduction. After employing
three different feature selections to best pick the top three most important features, and three
different supervised learning classifiers, we learned that SVC RBF was the best classifier for red
wine with the accuracy score of 85.6% and LinearSVC (Linear) was the best for white wine with
the accuracy score of 83.5%. This dataset was very difficult to work with because of the
imbalance of data issue in quality feature column, and the subjectiveness of the grading.
