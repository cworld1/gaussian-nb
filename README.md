# Simulating the Parsimonious Bayesian Algorithm with Python and Finding the Probability Prediction of Breast Cancer

The full probability formula and Bayes' formula are important formulas in probability theory, mainly used to calculate the probability of more complex events, and they are essentially a combination of the addition formula and the multiplication formula. Probability theory and mathematical statistics is a mathematical discipline that studies the statistical regularity of random phenomena and originated in the 17th century. It has developed into many fields of science and society. The two formulas have been studied in many countries from the 17th century to the present in many ways.

One of the important topics of probability theory is the desire to derive the probability of unknown complex events from the known probability of simple events. To achieve this, a complex event is often divided into a number of mutually incompatible events, and then the final result is obtained by calculating the probabilities of these simple events separately and finally using the additivity of probabilities. This is the basic idea of the full probability formula. Putting the above in order is the full probability formula. The full probability formula is a very important basic formula in probability theory. Through the study of probability theory courses, it is found that there are multiple tolerances that can be further deepened and mined to obtain a broader, more concise, and more practical conclusion to enrich and improve the theoretical system of probability theory. It provides an effective way to calculate the probability of complex events and simplifies the problem of calculating the probability of a complex event. It plays a very important role in probability theory, and the flexible use of the full probability formula will bring us great convenience in solving problems. The full probability formula contains the mathematical ideas of turning whole into zero and complexity into simplicity; the essence of the full probability formula: $P(B)$ in the full probability formula is an average probability, which is the weighted average of the conditional probabilities $P(B|A_i)$, where the weight added to each conditional probability is the probability of the occurrence of the event $A_i$ as a condition.

The Bayesian formula first appeared in a work by the English scholar T. Bayes (1702-1761) in 1763, after his death. In terms of formal derivation, this formula is bland, it is just a simple derivation of the definition of conditional probability and the full probability formula. What makes it famous is its realistic and even philosophical interpretation: a situation that was thought to be not very likely can become very likely by the occurrence of a certain event; or, on the contrary, Bayes' formula quantitatively The Bayesian formula portrays this change quantitatively.

## Principle chapter

### Conditional probability

Conditional probability is the probability that something will happen under certain conditions. For example, the probability of a man confessing his love to a girl is 20%, which is written as $P(A)$ = 20%, where A stands for "the man confesses his love to the girl". If the man drives a Mercedes-Benz, the probability of a successful confession to a girl is 50%, then $P(A|B)$ = 50%, where B stands for "the man drives a Mercedes-Benz". After all, girls do prefer the grand scene. This $P(A|B)$ is the conditional probability.

### Joint probability

What is the joint probability that the man drives a Mercedes and has a 1% chance of confessing his love to the girl, then write $P(AB)$ = 1%. You may wonder, isn't the probability of a successful confession 20%? Isn't the joint probability as high as 50%? Why is the joint probability so low? This may be because the man is particularly poor, the probability of driving a Mercedes is really low so dragged down the joint probability.

### Difference and connection between conditional probability and joint probability

To summarize, conditional probability is the probability of event A occurring under the condition of B. And the joint probability is the probability that the event A and B occur simultaneously. The two are related and can be expressed by the following formula:

$$
P(AB) = P(A|B)P(B) = P(B|A)P(A)
$$

That is, the joint probability is equal to the conditional probability multiplied by the probability of the condition.
When $A$ and $B$ are independent events, for example, Michelle bought a tape one day in 2001 and at the same time China's Olympic bid was successful, these two events can be considered as independent events.

$$
P(AB) = P(A|B)P(B) = P(A)P(B)
$$

### Full probability formula

The problem is more complicated if the event $B$ is not a condition, but a set of conditions that are mutually exclusive and exhaustive of all possibilities. For example, a boy confesses his love to a girl during the day and courts her at night. The confession is recorded as event $A$, the daytime as event $B_1$, and the nighttime as event $B_2$. It is not difficult to obtain:

$$
P(A) = P(AB1) + P(AB2) = P(B1|A)P(A) + P(B2|A)P(A) = P(A|B1)P(B1) + P(A|B2)P(B2)
$$

Its general form is:

$$
P(A) = \normalsize\sum_{1}^{m}P(A|B_{i})P(B_{i})
$$

### Bayesian formula

The name of the algorithm, "plain Bayesian", comes from the Bayesian formula. To make it easier to understand, instead of using the event $A$ and the event $B$, we use the common $x, y$ for machine learning.

$$
P(y_{i}|x) = P(x|y_{i})P(y_{i})/P(x)
$$

Expanding the denominator $P(A)$ according to the full probability formula, we can obtain:

$$
P(y_{i}|x) = P(x|y_{i})P(y_{i})/\sum_{1}^{m}P(x|y_{j})P(y_{j})
$$

That is to say, if the feature $x$ is known and you want to solve for $y_i$, you only need to know the prior probability $P(y_i)$, and the likelihood $P(x|y_i)$ to solve for the posterior probability $P(y_i|x)$. And for the same sample $x$, $P(x)$ is a constant and can be left out of the calculation.

In the full probability formula and Bayesian formula, $y_1, y_2, y_3, ... , y_n$ are the various causes that accompany the occurrence of the outcome $x$, and $P(y)$ is the probability of the occurrence of the various causes, which is generally given empirically and is called the prior probability. $P(y_i|x)$ reflects the new result of the probability of occurrence of various conditions after the experiment and can be used to correct $P(y_i)$. The full probability formula is used for "from cause to effect" and the Bayesian formula is used for "from effect to cause".

### Gaussian distribution

If $x$ is a continuous variable, how to estimate the likelihood $P(x|y_i)$? We can assume that $x$ obeys the Gaussian distribution (normal distribution) conditional on $y_i$. The $P(x|y_i)$ can be calculated from the probability density function of the normal distribution with the following equation.

$$
P(x) = \large\frac{1}{\sigma\sqrt{2\pi}}\normalsize e^{-\frac{(x-\mu)^{2}}{2\sigma^2{}}}
$$

### Gaussian Parsimonious Bayes

If $x$ is multidimensional data, then we can assume that $P(x_1|y_i),P(x_2|y_i)\ ... \ P(x_n|y_i)$ correspond to events that are independent of each other, and these values are multiplied together to obtain $P(x|y_i)$, "independent of each other" is the simplicity of plain Bayes.

## Realization chapter

### Create GaussianNB class

Initialize and store the prior probabilities, the mean and variance of the training set and the number of categories of the label.

```Python
class GaussianNB:
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None
```

### Calculate prior probabilities

Calculate the percentage of each category with Python's own Counter, and store the results in a numpy array.

```Python
def _get_prior(label: ndarray)->ndarray:
    cnt = Counter(label)
    prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
    return prior
```

### Calculate the training set mean

Calculate the mean value for each label category separately.

```Python
def _get_avgs(self, data: ndarray, label: ndarray)->ndarray:
    return np.array([data[label == i].mean(axis=0) for i in range(self.n_class)])
```

### Calculate training set variance

The variance is calculated separately for each label category.

```Python
def _get_vars(self, data: ndarray, label: ndarray)->ndarray:
    return np.array([data[label == i].var(axis=0) for i in range(self.n_class)])
```

### Calculating the likelihood

The likelihood is calculated from the probability density function of the Gaussian distribution and then multiplied together to obtain the likelihood.

```Python
def _get_likelihood(self, row: ndarray)->ndarray:
    return (1 / sqrt(2 * pi * self.vars) * exp(
        -(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)
```

### Training Model

```Python
def fit(self, data: ndarray, label: ndarray):
    self.prior = self._get_prior(label)
    self.n_class = len(self.prior)
    self.avgs = self._get_avgs(data, label)
    self.vars = self._get_vars(data, label)
```

### Predicting prob

The prob of each label is obtained by multiplying the prior probability by the likelihood and normalizing it.

```Python
def predict_prob(self, data: ndarray)->ndarray:
    likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=data)
    probs = self.prior * likelihood
    probs_sum = probs.sum(axis=1)
    return probs / probs_sum[:, None]
```

### Predict label

For a single sample, the category corresponding to the maximum value of prob is taken, which is the predicted value of label.

```Python
def predict(self, data: ndarray)->ndarray:
    return self.predict_prob(data).argmax(axis=1)
```

## Effect evaluation

### main function

Use a well-known breast cancer dataset, split into training and test sets in the ratio of 7:3, train the model, and count the accuracy.

```Python
def main():
    print("Tesing the performance of Gaussian NaiveBayes...")
    data, label = load_breast_cancer()
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=100)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    y_hat = clf.predict(data_test)
    acc = _get_acc(label_test, y_hat)
    print("Accuracy is %.3f" % acc)
```

### Effect show

ACC 0.942, running time 11 milliseconds.

```Bash
python -u "d:\Project\Python-Project\Gaussian-NB\main.py"
#> Tesing the performance of Gaussian NaiveBayes...
#> Accuracy is 0.942
#> Total run time is 11.2 ms
```

## Summary

This paper describes in detail the full probability formula, the applications of the full probability formula, the generalization of the full probability formula and its applications, the Bayesian formula and its connection to the full probability formula, and the generalization theorem of the Bayesian formula and its applications. Through these detailed accounts, it can be seen that the applications of the two probability formulas are multifaceted. The extension of the two probability formulas will further expand the use of the two probability formulas and become an effective tool for us to solve more complex problems. However, due to the short research period, there are still many shortcomings in this study, and this paper only gives a few examples to illustrate their applications, and in fact their applications go far beyond that, and can be used to solve a series of uncertain problems such as investment, insurance, and engineering. What other kinds of problems should be solved with full probability formulas? What kind of problems should be solved with Bayesian formulas? What kind of problems should be solved by combining two formulas? In what kind of problems should the full probability formula or Bayesian formula be applied in several specific steps to solve? None of the paper has come up with a clear method and classification, and these are the issues to be further studied in depth in the future. With the rapid development of society and the increasingly fierce competition in the market, decision makers have to make comprehensive judgments by examining past information and the current situation, and the discipline of probabilistic analysis for decision making has become increasingly important. The use of mathematical methods to quantitatively correlate medical problems gives credibility to their conclusions and facilitates the treatment of patients. In conclusion the proper application of these two probability formulas and their extended forms helps to further study the probability of the target event and its conditions for each induced event in the experiment of multiple stochastic processes, helps to grasp the interactions between random events, provides more valuable decision information for production practice, and then becomes an effective tool for our problem solving.

## License

The MIT License.
