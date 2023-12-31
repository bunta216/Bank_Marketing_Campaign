# Project - Bank Marketing Campaign

In today's competitive marketing world, banks come up with many packages to attract customers. It would be helpful for the banks to design a package based on a certain demographic, so that they can specifically target those customer groups or customize their packages accordingly. The goal of this classification model is to predict if a customer would buy their term deposit or not. Where a term deposit is a deposit that a financial institution offers with a high fixed rate and a fixed maturity date. This is for a Portuguese banking institution.

### Input Data 
Features of this dataset includes:
* Age
* Job type
* Marital status
* Education
* Credit default status
* Average yearly balance
* Mortgage
* Personal loan
* Contact type
* Day of the month
* Month of the year
* Call duration
* Number of contacts
* Days since last contact
* Number of contacts
* Previous campaign outcome
* Output: Opened Term Deposit

### Machine Learning Model Process

![](img/process.png)


Using Pipeline and GridSearchCV with five folds we built four models:
* K-Nearest Neighbor (KNN)
* Support Vector Machine (SVC)
* Random Forest (RFC)
* Gradient Boosting (GBC)


### Imbalanced data

The data sets are imbalanced, with 88% no and 12% yes. Balanced data set is created for training and comparing to the imbalanced data set.  

### Comparison of models and Results:

#### Results baseline (unbalanced) classes

| Classifier | Best Parameters | | Precision | Recall | Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| KNN  | neighbors: 5, pca components: 28 |  <ul><li>0</li><li>1</li></ul> |  <ul><li>91%</li><li>49%</li></ul> |  <ul><li>97%</li><li>24%</li></ul>  |88.88%  |
| SVM  | pca components: 30, C: 1, gamma: 1e-07, kernel: linear |  <ul><li>0</li><li>1</li></ul> |  <ul><li>91%</li><li>61%</li></ul> |  <ul><li>98%</li><li>26%</li></ul>  |89.48%  |
| Random Forest Classification  | pca components: 34, estimators: 200 |  <ul><li>0</li><li>1</li></ul> |  <ul><li>91%</li><li>58%</li></ul> |  <ul><li>97%</li><li>28%</li></ul>  |90.42%  |
| Gradient Boosting Classifier   | learning rate: 0.1, pca components: 39, estimators: 200 |  <ul><li>0</li><li>1</li></ul> |  <ul><li>92%</li><li>55%</li></ul> |  <ul><li>96%</li><li>36%</li></ul>  |90.45%  |


#### Results balanced classes

| Classifier | Best Parameters | | Precision | Recall | Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| KNN  | neighbors: 5, pca components: 28 |  <ul><li>0</li><li>1</li></ul> |  <ul><li>75%</li><li>80%</li></ul> |  <ul><li>80%</li><li>75%</li></ul>  |78.51%  |
| SVM  | pca components: 30, C: 1, gamma: 1e-07, kernel: linear |  <ul><li>0</li><li>1</li></ul> |  <ul><li>81%</li><li>83%</li></ul> |  <ul><li>82%</li><li>81%</li></ul>  |82.20%  |
| Random Forest Classification  | pca components: 34, estimators: 200 |  <ul><li>0</li><li>1</li></ul> |  <ul><li>83%</li><li>83%</li></ul> |  <ul><li>82%</li><li>84%</li></ul>  |83.36%  |
| Gradient Boosting Classifier   | learning rate: 0.1, pca components: 39, estimators: 200 |  <ul><li>0</li><li>1</li></ul> |  <ul><li>83%</li><li>83%</li></ul> |  <ul><li>83%</li><li>83%</li></ul>  |84.50%  |

### Conclusion:
Gradient Boosting Classifier outperformed all other classifiers.

Balancing classes reduced accuracy but also reduced overfitting and increased recall.
