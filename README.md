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

### Multi-classifier

### One-vs-all

### Mathematics


---

## Temporary dump of information

A multi-classifier simply is (I think), many binary classifiers that each answer to the question *"is this object part of class `x`"*?

In this case, it would be 4 classifiers that each answer to *"is this student part of `Gryffindor | Hufflepuff | Ravenclaw | Slytherin`"*?

I suppose that we would give a student to each of the 4 classifiers, each one would give a probability of the student being part of their Hogwarts house, e.g. `Gryffindor: 0.8, Hufflepuff: 0.1, Ravenclaw: 0.05, Slytherin: 0.05`, and we would tell all of them, *"actually, it's `Gryffindor: 0, Hufflepuff: 0, Ravenclaw: 0, Slytherin: 1`"*, for each student.

---

# Resources

- [📺 Multiclass - One-vs-rest classification](https://www.youtube.com/watch?v=EYXSve6T5BU)
- [📺 Logistic Regression Cost Function | Machine Learning | Simply Explained](https://www.youtube.com/watch?v=ar8mUO3d05w)
