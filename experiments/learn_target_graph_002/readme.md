## Learning-target-2

Here I'm asking the dags to learn the add function on two inputs, but I'm requiring them to use at least five intermediate neurons to do so -- want to see if they can 
learn to forget neurons when necessary. 

## Expt 697

![](plots/700/learned_graph_2.png)

They really seem to love the .5 scaling activation function!
This one had a score of -.3, highest in the batch.


## Notes
* One common failure mode seems to be the network 'giving up' and learning to make no connections at all. See, for example, 698 and 701. 
* I wonder if I'm using the right cost function for this task. Right now I'm training with a negative mean-squared-error -- which is minimized when the distribution learns to perform the desired task, but perhaps leads to quite flat loss landscapes otherwise... worth thinking about. 

## Results
* Training overly-long graphs seems difficult and highly seed-dependent.