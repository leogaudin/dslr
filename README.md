<h1 align='center'>🎩 dslr</h1>

> ⚠️ This guide is written assuming that you have done `ft_linear_regression` (if not, do it), and does not go in depth on things seen in the `Python Piscine for Data Science`, like `pandas`, `numpy`, `matplotlib`, etc.

## Table of Contents

- [Introduction](#introduction) 👋
- [Decrypting the subject](#decrypting-the-subject) 🔍
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

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

This is the sigmoid function we talked about earlier, the blue line representing the probability of passing the exam.

![Sigmoid curve](./assets/sigmoid_curve.gif)

> *Here is how the sigmoid curve changes based on the value of `z`. The higher `z` is, the steeper the curve is, i.e. there is a threshold where the probability goes from 0 to 1 almost instantly.*

Then, we have the hypothesis function:

$$
h_{\theta}(x) = g(\theta^T x)
$$

$h_{\theta}(x)$ is the hypothesis we are making based on the input $x$ and the weights $\theta$ passed to $g(z)$.

We just learned what was this $g(z)$, but what about $\theta^T x$?

The $T$ in $\theta^T$ means "transpose". This implies that we are working with vectors.

Indeed, we have many parameters in our dataset, not a single one like in `ft_linear_regression`.

Our variables should then look like this:

$$
\theta = \begin{bmatrix}
w_{\text{param1}} \\
w_{\text{param2}} \\
w_{\text{param3}} \\
... \\
\end{bmatrix}
$$

$$
x = \begin{bmatrix}
\text{param1} \\
\text{param2} \\
\text{param3} \\
... \\
\end{bmatrix}
$$

It is simply a way of reprensenting the weights and the input in a more compact way, the same way we had $y = \theta_0 + \theta_1 x$ in `ft_linear_regression`.

---

## Temporary dump of information

A multi-classifier simply is (I think), many binary classifiers that each answer to the question *"is this object part of class `x`"*?

In this case, it would be 4 classifiers that each answer to *"is this student part of `Gryffindor | Hufflepuff | Ravenclaw | Slytherin`"*?

I suppose that we would give a student to each of the 4 classifiers, each one would give a probability of the student being part of their Hogwarts house, e.g. `Gryffindor: 0.8, Hufflepuff: 0.1, Ravenclaw: 0.05, Slytherin: 0.05`, and we would tell all of them, *"actually, it's `Gryffindor: 0, Hufflepuff: 0, Ravenclaw: 0, Slytherin: 1`"*, for each student.

---

# Resources

- [📺 Multiclass - One-vs-rest classification](https://www.youtube.com/watch?v=EYXSve6T5BU)
- [📺 Logistic Regression Cost Function | Machine Learning | Simply Explained](https://www.youtube.com/watch?v=ar8mUO3d05w)
- [📖 Multi-Class Classification: One-vs-All](https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5)
