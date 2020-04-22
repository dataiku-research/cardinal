Uncertainty based query sampling
================================

[Settles2009]_ has laid the fundations of formal active learning. Uncertainty
methods are described as the most basic ones. They assume that the most
informational samples lie on the decision boundary of the classifier, where the
confidence score of the model is the lowest. These methods are therefore called
uncertainty sampling methods.

Three strategies are commonly applied although most papers tend to call them all
"uncertainty sampling".

.. [Settles2009] Settles, B. (2009). Active learning literature survey. University
   of Wisconsin-Madison Department of Computer Sciences.

Notations
---------

In the following, we assume a classification problem among :math:`k` classes. For 
a given sample :math:`x` we define as :math:`p(c_i|x)` its probability to belong
to the :math:`i`<sup>th</sup> highest class:

.. math::

   \forall  i,j \in [1 .. k]^2, \quad i > j \Rightarrow P(c_i|x) > P(c_j|x) 


Least confidence
----------------

The least confidence sampling selects the samples for which the maximum probability
among all classes is the lowest. It amounts to selecting the samples for which the
model is not confident about the class it attributed:

.. math::

   lc(x) = 1 - P(c_1|x)

This method seems the most natural however, in multi-class cases, the score of
the majority class may not be the most significant indicator. For example, the
model may struggle between two classes.


Smallest margin sampling
------------------------

Smallest margin sampling takes into account the difference of prediction between the first
and second classes chosen by the model. This method is particularly useful when we
expect the samples to be tied between two classes. For example, in MNIST, 3 and 5
look alike, and we may expect some samples to be in-between:

.. math::

   sm(x) = 1 - (P(c_1|x) - P(c_2|x))


Note that in some papers that using support-vector-machine based classifiers, the term
margin can be used to designate the decision boundary and so margin sampling often refers
to least confidence sampling.

Entropy sampling
----------------

In the end, the real sense of uncertainty method is to determine how much information
the model has on our sample. Information theory has used Shannon’s entropy since ages
to measure the amount of information in data. The last method proposes to use it on
the prediction probabilities where more entropy means more chances to be selected.

.. math::

   es(x) = H(c)

Be careful though: sometimes, classes in the tail of the prediction does not make sense.
In that case, entropy sampling may be misled.
