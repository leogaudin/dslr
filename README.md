<h1 align='center'>🎩 dslr</h1>

> ⚠️ This guide is written assuming that you have done `ft_linear_regression` (if not, do it), and does not go in depth on things seen in the `Python Piscine for Data Science`, like `pandas`, `numpy`, `matplotlib`, etc.

## Table of Contents

- [Introduction](#introduction) 👋
- [Decrypting the subject](#decrypting-the-subject) 🔍
	- [Logistic Regression](#logistic-regression) 📈
	- [Multi-classifier](#multi-classifier) 🔢
	- [One-vs-all](#one-vs-all) 🏁
	- [Mathematics](#mathematics) 🧮
		- [Sigmoid function](#sigmoid-function) 📈
		- [Hypothesis function](#hypothesis-function) 🤔
		- [Cost function](#cost-function) 💰
		- [Derivative of the cost function](#derivative-of-the-cost-function) 📉
- [Resources](#resources) 📖

## Introduction

With `dslr`, we take a step further into the world of data science.

We are given a dataset of students at Hogwarts, with their grades in various subjects, and their house.

The final goal is to predict the house of a student based on similar data.

The subject is divided into the following parts:

1. **Data Analysis**
2. **Data Visualization**
	1. Histogram
	2. Scatter plot
	3. Pair plot
3. **Logistic Regression**

Parts 1 and 2 honestly do not require a tutorial. The first part is just reading the data and doing some basic statistics, and the second part is just plotting the data into Matplotlib (see appendix in the subject)

The third part is where the real work is done, and where this guide will focus.

## Decrypting the subject

Let's first recap all the cryptic terms used in the subject:

- Logistic Regression
- Multi-classifier
- One-vs-all

And worst of all, appendix VIII.1, *Mathematics*:

![appendix](./assets/appendix.webp)

> *What is this supposed to mean?*

### Logistic Regression

Logistic Regression is a classification algorithm used to tell if an object is part of a class or not.

Unlike linear regression which takes a scalar input and gives a scalar output (e.g. `price = mileage * weight`), logistic regression gives a probability of the input being part of a class (e.g. `[0, 1] = input * weight`).

![Sigmoid Example](./assets/sigmoid_example.webp)

> *Here, the sigmoid function gives the probability of a student passing their exam based on the number of hours they studied.*

### Multi-classifier

Multi-classification is simply when you have more than 2 classes to classify.

Instead of having a binary class like "obese" or "not obese", you have multiple classes like `Gryffindor`, `Hufflepuff`, `Ravenclaw`, or `Slytherin`.

That is where one-vs-all comes in.

### One-vs-all

The one-vs-all strategy is a way to apply the logic we just described to the problem of having multiple classes.

Let's take the example from [this lecture](https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5):

> Suppose you have classes `A`, `B`, and `C`. We will build one model for each class:
>
> - Model 1: `A` or `BC`
> - Model 2: `B` or `AC`
> - Model 3: `C` or `AB`
>
> Another way to think about the models is each class vs everything else (hence the name):
>
> - Model 1: `A` or not `A`
> - Model 2: `B` or not `B`
> - Model 3: `C` or not `C`

In our case, we will have 4 models:

- Model 1: `Gryffindor` or not `Gryffindor`
- Model 2: `Hufflepuff` or not `Hufflepuff`
- Model 3: `Ravenclaw` or not `Ravenclaw`
- Model 4: `Slytherin` or not `Slytherin`

### Mathematics

Now let's dive into this appendix, starting with the last equation before the derivative:

#### Sigmoid function

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

This is the sigmoid function we talked about earlier, the blue line representing the probability of passing the exam.

![Sigmoid curve](./assets/sigmoid_curve.gif)

> *Here is how the sigmoid curve changes based on the value of `z`. The higher `z` is, the steeper the curve is, i.e. there is a threshold where the probability goes from 0 to 1 almost instantly.*

#### Hypothesis function

$$
h_{\theta}(x) = g(\theta^T x)
$$

$h_{\theta}(x)$ is the hypothesis we are making based on the input $x$ and the weights $\theta$ passed to $g(z)$.

We just learned what was this $g(z)$, but what about $\theta^T x$?

The $T$ in $\theta^T$ means "transpose", which is a vector operation.

Indeed, we have many parameters in our dataset, not a single one like in `ft_linear_regression`. We therefore need to "group" them in a vector.

> 💡 If you arrived here right after C, think of it as a 1-D array.
>
> If you are not familiar with vectors and matrices, you should do the `matrix` project, that can bring you very interesting bases for `dslr`.

Assuming that $\theta$ and $x$ are both column vectors as follows:

$$
\theta = \begin{bmatrix}
w_{\text{param1}} \\
w_{\text{param2}} \\
w_{\text{param3}} \\
\ldots
\end{bmatrix}
$$

$$
x = \begin{bmatrix}
\text{param1} \\
\text{param2} \\
\text{param3} \\
\ldots
\end{bmatrix}
$$

"Multiplying" them as they are, with an element-wise product for example, would result in a third vector that would look like:

$$
\theta x = \begin{bmatrix}
w_{\text{param1}} \times \text{param1} \\
w_{\text{param2}} \times \text{param2} \\
w_{\text{param3}} \times \text{param3} \\
\ldots
\end{bmatrix}
$$

The notation $\theta^T x$ is a way of clarifying we are doing a dot product between the two vectors, which would look like:

$$
\theta^T x = w_{\text{param1}} \times \text{param1} + w_{\text{param2}} \times \text{param2} + w_{\text{param3}} \times \text{param3} + \ldots
$$

This operation gives us a scalar (i.e. a single value), which is what we want to pass to the sigmoid function.

#### Cost function

#### Derivative of the cost function

# Resources

- [📺 YouTube − Multiclass - One-vs-rest classification](https://www.youtube.com/watch?v=EYXSve6T5BU)
- [📺 YouTube − Logistic Regression Machine Learning Example | Simply Explained](https://www.youtube.com/watch?v=U1omz0B9FTw)
- [📺 YouTube − Logistic Regression Cost Function | Machine Learning | Simply Explained](https://www.youtube.com/watch?v=ar8mUO3d05w)
- [📖 Rice University − Multi-Class Classification: One-vs-All](https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5)
- [💬 Stack Exchange − Theta * X vs Sum_j=1(Theta_j * x_j)](https://math.stackexchange.com/questions/3485981/thetatx-vs-sum-j-1n-theta-j-x-j)
- [💬 Stack Exchange − Theta transposes to x](https://math.stackexchange.com/questions/60212/theta-transposes-to-x)
- [📖 Wikipedia − Dot product](https://en.wikipedia.org/wiki/Dot_product)
