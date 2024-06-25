# leaffliction
42 project about identifying leaves

## CNN
Following [this keras tutorial by Victor Zhou](https://victorzhou.com/blog/keras-cnn-tutorial/), we implemented a basic CNN with one Conv2D, one MaxPolling and some fully-connected layers before the softmax.<br>
This CNN could easily be improved as we slightly overfit on some models. Adding a Drop layer should stop overfitting.<br>More layers (activation of ) will also improve accuracy but will take more time for training.<br><br>
We chose not to go further for the CNN as we already reached the 90% accuracy needed to validate the project (94% with hard voting).
## voting algorithms
model1_pred = [0.9, 0.5, 0.002]<br>
model2_pred = [0.1, 0.8, 0.003]<br>
model3_pred = [0.1, 0.8, 0.003]

### soft_vote
avg for each class then pick best
soft_vote_avg = [0.37, 0.7, 0.0027]
soft_vote_pred = 1

### hard_vote
pick best out of all pred<br> 
hard_vote_pred = 0