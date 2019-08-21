## Learn connection count

Goal: teach the model to only output graphs with a specific number of nonzero connections.

### Observations
* If the graph batch size is set to 1 (714, 715) the model doesn't seem to learn at all
* With a batch size of 10, I do get learning (716: target 10)