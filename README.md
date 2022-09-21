# Logistic-Regression-from-scratch
Numpy implementation of Logistic Regression (w/o explicit ML libraries, such as Pytorch or Tensorflow)

## What is Logistic Regression
### Logistic Function
- also called as <b>Sigmoid Function</b></li>
- $y$ = $1 \over 1+e^{-x}$
<img src="https://t1.daumcdn.net/cfile/tistory/275BAD4F577B669920" width="250" height="250"></img>
- Output of logistic function is always in (0,1)  ==> Logistic function satisfies the condition of <b>Probability Density Function</b>

### Odds ( $\ne$ Probability)
- defined as 'probability that the event will occur divied by the probability that the event will not occur <br>
<img src="imgs/odd-equation.PNG" width="250" height="90"></img>
<img src="https://miro.medium.com/max/1400/1*8ix_A7GUKH9AsZxouYg-uw.png" width="300" height="150"></img>

### Mapping the linear equation f(x) to log odds
- In Linear Regression, $f(x) = W^TX$
- Let p = P(y=1|x)<br>
 <img src="imgs/log-odds.PNG" width="250" height="70"></img>
- Take exponents for both side:<br>
 <img src="imgs/take-exponent.PNG" width="250" height="70"></img>
- Then, p (the probability of y=1 for x) is <br>
 <img src="imgs/p-definition.PNG" width="250" height="70"></img>

### Loss Function Formulation, E( $W$ )
 <img src="imgs/loss-formula.PNG" width="270" height="70"></img>



## Sample Dataset
#### EMNIST
<img src="http://greg-cohen.com/datasets/emnist/featured.png" width="250" height="200"></img>
#### Breast Cancer
<img src="https://pyimagesearch.com/wp-content/uploads/2019/02/breast_cancer_classification_dataset.jpg" width="300" height="200"></img>

## Reference
SungKyunKwan University, College of Computing, SWE3050_41
