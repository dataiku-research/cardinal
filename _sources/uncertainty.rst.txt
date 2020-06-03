Uncertainty-based Query Sampling
================================

[Settles2009]_ has laid the fundations of formal active learning. Uncertainty
methods are described as the most basic ones. They assume that the most
informational samples lie on the decision boundary of the classifier, where the
confidence score of the model is the lowest. These methods are therefore called
uncertainty sampling methods.

Three strategies are commonly applied although most papers tend to call under the
umbrella term "uncertainty sampling".

.. [Settles2009] Settles, B. (2009). Active learning literature survey. University
   of Wisconsin-Madison Department of Computer Sciences.

Notations
---------

In the following, we assume a classification task among :math:`k` classes. For 
a given sample :math:`x`, we order the :math:`k` classes such that :math:`p(c_i|x)`
denotes its probability of belonging to the :math:`i`-th highest class.
In other words, we have:

.. math::

   \forall  i,j \in [1 .. k]^2, \quad i > j \Rightarrow P(c_i|x) > P(c_j|x) 

This ordering of the classes depends on the sample :math:`x` and it is only used for
simplicity of notations.

Least Confidence
----------------

Least confidence sampling selects the samples for which the maximum probability
among all classes is the lowest. It amounts to selecting the samples for which the
model is not confident on its class attribution:

.. math::

   lc(x) = 1 - P(c_1|x)

If this method seems the most natural, in multi-class cases, the score of
the majority class may not be the most significant indicator. For example, the
model may struggle between two classes.


Smallest Margin Sampling
------------------------

Smallest margin sampling takes into account the difference of prediction between the first
and second classes chosen by the model. This method is particularly useful when we
expect the samples to be tied between two classes. For example, in MNIST, 3 and 5
look alike, and we may expect some samples to fall in-between:

.. math::

   sm(x) = 1 - (P(c_1|x) - P(c_2|x))


Note that in some papers using support-vector-machine based classifiers, the term
margin can be used to designate the decision boundary and so margin sampling often refers
to least confidence sampling.

Entropy Sampling
----------------

In the end, the real sense of uncertainty method is to determine how much information
the model has on our sample. Information theory classicaly uses Shannon’s entropy
to measure the amount of information in data. Entropy sampling select samples with the highest predicted probabilities entropy.

.. math::

   es(x) = H(c) = -\sum_{i}P(c_i|x) \log(P(c_i|x))

Be careful though: sometimes, classes in the tail of the prediction does not make sense.
In that case, entropy sampling may be misleading.
