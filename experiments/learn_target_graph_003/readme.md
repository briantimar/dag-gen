### Learning target 3
Trying to learn a more complicated computational graph, by minimizing the MSE over inputs.

Observations:
* The score tends to plateau rather quickly
* The graph distribution keeps settling down to a size that's *too small* (zero or one intermediate nodes, and sometimes no connections)
* If I force the graphs to use two intermediate vertices (708), they just set those to bias units. weird
* If I force the graphs to use two intermediate vertices, *and* to use only the subset of activations that I used to construct the target function (709, 710, 711), they learn to make no target connections at all! Even weirder