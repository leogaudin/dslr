<h1 align='center'>🎩 dslr</h1>

## Temporary dump of information

A multi-classifier simply is (I think), many binary classifiers that each answer to the question *"is this object part of class `x`"*?

In this case, it would be 4 classifiers that each answer to *"is this student part of `Gryffindor | Hufflepuff | Ravenclaw | Slytherin`"*?

I suppose that we would give a student to each of the 4 classifiers, each one would give a probability of the student being part of their Hogwarts house, e.g. `Gryffindor: 0.8, Hufflepuff: 0.1, Ravenclaw: 0.05, Slytherin: 0.05`, and we would tell all of them, *"actually, it's `Gryffindor: 0, Hufflepuff: 0, Ravenclaw: 0, Slytherin: 1`"*, for each student.

## Table of Contents

- [Resources](#resources) 📖

# Resources

- [📺 Multiclass - One-vs-rest classification](https://www.youtube.com/watch?v=EYXSve6T5BU)
