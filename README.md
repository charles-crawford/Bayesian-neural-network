# Bayesian-neural-net-house-price-prediction

### Overview
Epistemic and aleatoric uncertainty are important concepts to account for in machine learning. 
Both can be quantified in Bayesian models. In this repo a Bayesian neural network is built and optimized with TensorFlow and 
TensorFlow Probability. 

### Code
TensorFlow hyperparameter search functionality is included in `hparam_search.py.` An AWS S3 bucket will be created, and
the search results will be put in the bucket at the end of the search. Also, the EC2 GPU instance you're running on will
shut down after the hyperparameter search finishes. To ensure the AWS resources shut down properly, make sure to have 
these environment variables set where you run the search:

1. AWS_REGION
2. AWS_ACCOUNT_ID
3. SNS_QUEUE_NAME

CAVEAT: I found that a large network will blow out your memory. So make sure to add more resources or set your 
hyperparameters to avoid this. 

I had to overwrite a method in the [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint ) 
class so that the best TensorFlow Probability model could be saved and then reloaded after training. To track the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) 
during training, I created a custom TensorFlow metric. I also created a custom loss function that takes into account the 
[negative log likelihood](https://medium.com/deeplearningmadeeasy/negative-log-likelihood-6bd79b55d8b6), 
[mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error),
[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient). Due to the magnitude
difference of the negative log likelihood and the other two, I added a scaling coefficient for each. These scaling 
coefficients can be added to the hyperparameter optimization search if so wanted. All of these modifications are in the
`utils/` directory.

### Data
I use the [Kaggle Ames, Iowa dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation)
for this work. As the topic for this repo is Bayesian neural nets, I used just basic data processing for the features. 
Feature engineering can definitely be employed to get better predictions. The processing is housed in a 
[ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
To that end, I created a couple custom transformers for use in the column transformer pipelines. 


### Helpful Links
[Keras BNN example](https://keras.io/examples/keras_recipes/bayesian_neural_networks/) </br>
[TF tensorboard hparam search](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams) </br>
[TF demo hparam search](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/hparams/hparams_demo.py) </br>
[Save a BNN workaround](https://github.com/tensorflow/probability/issues/325#issuecomment-477213850) </br>

### TODO
Apply the trainable prior techniques described below to further improve predictions.  </br>
[Regression with Probabilistic Layers in TensorFlow Probability](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) </br>
[Hierarchical Neural Network](https://twiecki.io/blog/2016/07/05/bayesian-deep-learning/) by Thomas Wiecki </br>