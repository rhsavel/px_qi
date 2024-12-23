Title:
Independent risk factors for less than “top box” doctor communication HCAHPS scores in an urban teaching hospital

Authors:
Richard Savel; Payam Benson; Carmen Collins; Srinivas Gongireddy; Christina Oquendo; Kwaku Gyekye; Eva Villar-Trinidad; Ije Akunyili

Introduction:
We wished to utilize data from the Press Ganey database to enhance the patient experience for our patients. The goal of this project was to determine the independent risk factors for patients giving us a less than a “top box” score on their Press Ganey/HCAHPS surveys in the domain “communication with doctors”.  We focused on the three questions:  

Question 1:  During this hospital stay, how often did doctors treat you with courtesy and respect?
Never, sometimes, usually, always.

Question 2:  During this hospital stay, how often did doctors listen carefully to you?
Never, sometimes, usually, always.

Question 3:  During this hospital stay, how often did doctors explain things in a way you could understand?
Never, sometimes, usually, always.

Methods:
We pulled data from all Press Ganey surveys from January 1 2023 to December 31 2023.  
The final list of potential risk factors for a less than “top box” HCAHPS score included: Age, gender, length of stay, discharge phone call, new medication during hospitalization, highest education level, language spoken at home, and zip code. A univariate logistic regression analysis was performed with a screening p value of < 0.2.  An ElasticNet analysis (combining Lasso and Ridge) was subsequently performed to minimize overfitting and collinearity.  A multivariate logistic regression analysis was then performed to determine the independent risk factors with a p of <0.05. A Hosmer-Lemeshow goodness-of-fit test was also performed.  All analysis was conducted using Python version 3.13.0 with relevant statistical libraries.  Missing data was managed using complete-case analysis.  Data was found to be non-normally distributed and is presented as median (25th-75th percentile).  Odds ratios are presented with 95% confidence intervals.  

Results: 
See attached data tables.  We analyzed data from a total of 803 unique patients.  

Conclusion:
Our study revealed that not receiving a discharge phone call was strongly associated with a nearly doubling of the probability of a patient giving a less than “top box” score for the doctor communication domain (with an odds ratio ranging from 1.67 for question 3 to 2.93 for question 1 and overall likelihood of giving a less than top box score increasing from 18% to 34% for patients who did not receive a post-discharge call).  This analysis can easily be customized for any medical center to help determine relevant action items for their specific patient population.  
