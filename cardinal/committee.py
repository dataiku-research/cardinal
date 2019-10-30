# Why recoding query by committee?
#
# This note motivates why the strategies have been recoded here.
#
# modAL: modAL is bagging oriented and forces to define learners in its
#        interfaces to use query by committee which is tedious
#
# libact: libact forces the use of bagging and also imposes to use its
#         abstractions.
#
# alipy: provides nice methods but does not return the confidence score.
#
# QBC is standby until we find a nice way to propose it in DSS
# (Not through a dedicated custom recipe)