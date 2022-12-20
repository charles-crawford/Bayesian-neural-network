import logging
import boto3
import h5py
import os
import json
import requests
import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OrdinalEncoder,
                                   PowerTransformer,
                                   OneHotEncoder,
                                   StandardScaler,
                                   RobustScaler,
                                   MinMaxScaler)

from utils.transformers.date_transformer import DateTransformer
from utils.transformers.log_transformer import LogTransformer
from .pearson_metric import PearsonMetric
from .tfp_checkpoint_callback import SaveTfpCheckpoint
from typing import Tuple, Dict, Optional, List
from collections.abc import Callable
from keras.engine.sequential import Sequential
from keras.callbacks import History
from tensorflow.python.data.ops.dataset_ops import BatchDataset

tfd = tfp.distributions
tfpl = tfp.layers

AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID')
SNS_QUEUE_NAME = os.getenv('SNS_QUEUE_NAME')


def get_preprocessor(numeric_transform: str) -> ColumnTransformer:
    """
    This method creates a generic column transformer object for preprocessing.


    Returns
    -------
    preprocessor : ColumnTransformer
        This returns a column transformer object with the transforms and
        imputators needed to get the data in machine readable format.

    """
    no_transform_features = ['OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                             'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

    binary_features = ['Street', 'CentralAir']
    date_features = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']

    ordinal_features = ['ExterQual', 'ExterCond', 'LandSlope', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                        'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                        'GarageCond', 'PoolQC']

    numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',
                        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                        'PoolArea', 'MiscVal']

    categorical_features = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig',
                            'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                            'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                            'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence',
                            'MiscFeature', 'MoSold', 'SaleType', 'SaleCondition']

    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]

    if numeric_transform == 'box-cox':
        numeric_steps.append(('numeric_scaler', PowerTransformer(method='box-cox')))
    elif numeric_transform == 'yeo-johnson':
        numeric_steps.append(('numeric_scaler', PowerTransformer(method='yeo-johnson')))
    elif numeric_transform == 'standard':
        numeric_steps.append(('numeric_scaler', StandardScaler()))
    elif numeric_transform == 'robust':
        numeric_steps.append(('numeric_scaler', RobustScaler()))
    elif numeric_transform == 'minmax':
        numeric_steps.append(('numeric_scaler', MinMaxScaler()))
    else:
        print('The numeric_transform value is not one of: box-cox, yeo-johnson, robust, or standard')
        print('Applying the robust scaler')
        numeric_steps.append(('numeric_scaler', RobustScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['Sev', 'Mod', 'Gtl'],
                                                       ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['NA', 'No', 'Mn', 'Av', 'Gd'],
                                                       ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                                                       ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                                                       ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                                                       ['NA', 'Fa', 'TA', 'Gd', 'Ex']
                                                       ],
                                           handle_unknown='use_encoded_value',
                                           unknown_value=-1))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    no_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    date_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('date_transformer', DateTransformer()),
        ('log_transformer', LogTransformer()),
        ('scaler', StandardScaler())
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('binary_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    transformers = [
        ('date', date_transformer, date_features),
        ('num', numeric_transformer, numeric_features),
        ('ordinal', ordinal_transformer, ordinal_features),
        ('no_transform', no_transform, no_transform_features),
        ('cats', categorical_transformer, categorical_features),
        ('binary', binary_transformer, binary_features),
    ]
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n) * 100
                )
            )
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfpl.VariableLayer(
                tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfpl.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def get_tensors(data: pd.DataFrame,
                feature_names: List[str],
                target_name: Optional[str] = None,
                batchsize: Optional[int] = 16) -> BatchDataset:

    numeric_features = data[feature_names].values

    if target_name:
        target = tf.cast(data[target_name].values, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target)).batch(batchsize)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(numeric_features).batch(batchsize)

    return dataset


def run_experiment(hparams: Dict,
                   features: List[str],
                   loss: Callable,
                   num_epochs: int,
                   train_dataset: tf.data.Dataset,
                   test_dataset: tf.data.Dataset,
                   tensor_board: Optional[bool] = False) -> Tuple[Sequential, History]:

    model = create_model(hparams, features, loss)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    save_tfp_model = SaveTfpCheckpoint('best-iowa-housing-price-model.h5',
                                       monitor='val_pearson',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False,
                                       verbose=0)

    callbacks = [save_tfp_model, early_stop]
    if tensor_board:
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./bnn-logs/')
        callbacks = [save_tfp_model, early_stop, tensor_board]
    print("Start training the model...")
    history = model.fit(train_dataset,
                        epochs=num_epochs,
                        validation_data=test_dataset,
                        callbacks=callbacks,
                        )

    print("Model training finished.")
    _, pearson, mse = model.evaluate(train_dataset, verbose=0)
    print(f"Train Pearson: {round(pearson, 3)}")
    print(f"Train MSE: {round(mse, 3)}")

    print("Evaluating model performance...")
    _, pearson, mse = model.evaluate(test_dataset, verbose=0)
    print(f"Test Pearson: {round(pearson, 3)}")
    print(f"Test MSE: {round(mse, 3)}")
    return model, history


def create_model(hparams: Dict, features: List[str], loss: Callable) -> Sequential:
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """
    n_layers = hparams['n_layers']
    n_units = hparams['n_units']
    learning_rate = hparams['learning_rate']
    activation = hparams['activation']
    dropout = hparams['dropout']
    batchsize = hparams['batchsize']

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(len(features),), name="input"))

    for i in range(n_layers):
        model.add(tfpl.DenseFlipout(n_units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout, trainable=True))

    model.add(tf.keras.layers.Dense(units=2))
    model.add(tfpl.IndependentNormal(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[PearsonMetric(), 'mean_squared_error'],
    )
    return model


def load_saved_model(params: Dict,
                     model_loc: str,
                     features: List[str],
                     loss: Callable) -> Sequential:

    model = create_model(params, features, loss)
    file = h5py.File(model_loc, 'r')
    weights = []
    for i in range(len(file.keys())):
        weights.append(file['weight' + str(i)][:])
    model.set_weights(weights)
    file.close()
    return model


def plot_loss(history: History):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('No. of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def kill_ec2(run_time):
    print('Killing the EC2.')
    sns_client = boto3.client('sns', AWS_REGION)
    ec2_client = boto3.client('ec2', AWS_REGION)
    message = {'training-status': 'completed', 'ec2-status': 'shut-down', 'run_time': run_time}
    instance_id = None

    try:
        token = requests.put("http://169.254.169.254/latest/api/token",
                             headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'}).content
        instance_id = requests.get("http://169.254.169.254/latest/meta-data/instance-id",
                                   headers={'X-aws-ec2-metadata-token': token}).text
        response_ec2 = ec2_client.stop_instances(InstanceIds=[instance_id])
        response_sns = sns_client.publish(TopicArn=f'arn:aws:sns:{AWS_REGION}:{AWS_ACCOUNT_ID}:{SNS_QUEUE_NAME}',
                                          Message=json.dumps(message),
                                          Subject='Training')
        print('Successfully sent to SNS')
        print('Successfully stopped instance id: {}'.format(instance_id))

    except Exception as e:
        print('Failed to stop instance id: {}'.format(instance_id))
        print('Error:'.format(repr(e)))
        response_sns = sns_client.publish(TopicArn=f'arn:aws:sns:{AWS_REGION}:{AWS_ACCOUNT_ID}:{SNS_QUEUE_NAME}',
                                          Message=json.dumps({'error-message': repr(e)}),
                                          Subject='Training')


def create_bucket(bucket_name):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        location = {'LocationConstraint': AWS_REGION}
        s3_client.create_bucket(Bucket=bucket_name,
                                CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def send_to_aws(results_file_name):
    try:
        bucket_name = 'bnn-hyperparam-search'
        create_bucket(bucket_name)
        upload_file(results_file_name, bucket_name)
    except Exception as e:
        print('error:', repr(e))
        logging.error(e)
