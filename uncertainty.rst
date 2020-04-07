Uncertainty based query sampling
================================

[Settles2009]_ summarizes the most common active learning methods and some advanced
ones. The most common ones are known under the term of uncertainty sampling as they
directly leverage the prediction scores of the classifier to determine the samples
for which the prediction is the less certain.

.. [Settles2009] Settles, B. (2009). Active learning literature survey. University
   of Wisconsin-Madison Department of Computer Sciences.

Note that a lot of papers report results under the name uncertainty sampling without
mentioning the method used. This appellation covers the three following.

In the following examples, we will consider a classification problem with 4 classes
for illustration purpose. The graph indicates the probability to be chose with regard
to the probability score among other classes.


Least confidence
----------------

The least confidence sampling takes the samples for which the maximum probability
among all classes is the lowest. It amounts to selecting the samples for which the
model is not certain about the class it attributed.

In this schema, the probability depends only on the distance to the closest category.

The problem of this method is that, in multi class cases in particular, the score of
the majority class may not be the most significant. We would also like to look like
at the rest of the classes.


Margin sampling
---------------

Margin sampling takes into account the difference of prediction between the first
and second classes chosen by the model. This method is particularly useful when we
expect the samples to be tied between two classes. For example, in MNIST, 3 and 5
look alike and we may expect some samples to be inbetween. 

It is clear in this schema, the probability to be chosen is the ratio between the
distance to the two closest classes.

But again, we may be in cases where there are more than two classes (3, 5 and 6?) and
in this case, we would like to take all the probabilites into account.

Entropy sampling
----------------

In the end, the real sense of uncertainty method is to determine how much information
the model has on our sample. Information theory has used Shannon’s entropy since ages
to measure the amount of information in data. The last method proposes to use it on
the prediction probabilities where more entropy means more chances to be selected.

Now the probability to be selected depends on the distance to all the other classes.

Be careful though: sometimes, classes in the tail of the prediction does not make sense.
In that case, entropy sampling may be misled.