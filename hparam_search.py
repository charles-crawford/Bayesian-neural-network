import os.path
import shutil
import boto3
import logging
import json
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from absl import flags, app
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from datetime import datetime as dt
from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.custom_loss import CustomLoss
from utils.pearson_metric import PearsonMetric
from utils.tfp_checkpoint_callback import SaveTfpCheckpoint
from utils.utils import (get_preprocessor,
                         create_bucket,
                         upload_file,
                         kill_ec2,
                         get_tensors,
                         prior,
                         posterior,
                         send_to_aws)

tfd = tfp.distributions
tfpl = tfp.layers

DATE = dt.strftime(dt.now(), '%Y-%m-%d-%H-%M-%S')
RESULTS_FILE_NAME = f'search-results-{DATE}.csv'

flags.DEFINE_integer(
    "num_session_groups",
    10,
    "The approximate number of session groups to create.",
)
flags.DEFINE_string(
    "logdir",
    f"logs-{DATE}/hparam_tuning",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)

df = pd.read_csv('house-prices-data/train.csv', index_col=0)
target_col = 'SalePrice'
y = df[target_col]

cols_to_drop = ['Utilities', target_col]
df.drop(cols_to_drop, inplace=True, axis=1)

print('shape:', df.shape)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=101)
pipeline = get_preprocessor('robust')

X_train = pipeline.fit_transform(X_train)
n_rows, n_cols = X_train.shape
features = [f'col_{i}' for i in range(n_cols)]
X_train = pd.DataFrame(X_train, columns=features)
print('shape:', X_train.shape)

y_train_scaler = StandardScaler()
y_train = y_train_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
X_train[target_col] = y_train

X_test = pipeline.transform(X_test)
X_test = pd.DataFrame(X_test, columns=features)
print('shape:', X_test.shape)

y_test_scaler = StandardScaler()
y_test = y_test_scaler.fit_transform(y_test.values.reshape(-1, 1)).flatten()
X_test[target_col] = y_test

HP_LAYERS = hp.HParam('layers', hp.IntInterval(1, 2))
HP_UNITS = hp.HParam('units', hp.IntInterval(1, 512))
HP_BATCHSIZE = hp.HParam('batchsize', hp.IntInterval(2, 256))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.0000001, 1.))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh', 'sigmoid', 'relu']))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.3))

hparams = [HP_LEARNING_RATE,
           HP_LAYERS,
           HP_UNITS,
           HP_DROPOUT,
           HP_BATCHSIZE,
           HP_ACTIVATION]

METRICS = [
    hp.Metric(
        "epoch_mean_squared_error",
        group="validation",
        display_name="mse (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_mean_squared_error",
        group="train",
        display_name="mse (train)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
]


def run_all(hparams, logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=hparams, metrics=METRICS)

    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups

    run_results = []

    for group_index in range(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform() for h in hparams}
        for repeat_index in range(sessions_per_group):
            run_id = f'run_{group_index}_{repeat_index}'
            session_id = str(session_index)
            session_index += 1
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )

                print("--- repeat #: %d" % (repeat_index + 1))
            history, val_pearson, val_mse, hparams_dict = run(
                base_logdir=logdir,
                session_id=session_id,
                hparams=hparams,
                run_id=run_id
            )

            run_results.append(
                dict(
                    run_id=run_id,
                    val_pearson=val_pearson,
                    val_mse=val_mse,
                    params=json.dumps(hparams_dict)
                )
            )

            # write every result while training in case of failure
            df_results = pd.DataFrame.from_records(run_results)
            df_results.to_csv(results_file_name)


def run(base_logdir, session_id, hparams, run_id):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    model, hparams_dict = model_fn(hparams=hparams)
    logdir = os.path.join(base_logdir, session_id)

    print('\nhparams:')
    pprint(hparams_dict)

    hparams_callback = hp.KerasCallback(logdir, hparams)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_pearson', patience=50)
    tensor_board = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq=flags.FLAGS.summary_freq,
        profile_batch=0,  # workaround for issue #2084
    )

    batchsize = hparams[HP_BATCHSIZE]
    train_dataset = get_tensors(X_train, features, 'SalePrice', batchsize=batchsize)
    test_dataset = get_tensors(X_test, features, 'SalePrice', batchsize=batchsize)

    history = model.fit(train_dataset,
                        epochs=2000,
                        validation_data=test_dataset,
                        verbose=1,
                        shuffle=False,
                        callbacks=[hparams_callback, early_stop],
                        )
    _, val_pearson, val_mse = model.evaluate(test_dataset, verbose=0)
    return history, val_pearson, val_mse, hparams_dict


def model_fn(hparams):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    Returns:
      A compiled Keras model.
    """

    n_layers = hparams[HP_LAYERS]
    n_units = hparams[HP_UNITS]
    learning_rate = hparams[HP_LEARNING_RATE]
    activation = hparams[HP_ACTIVATION]
    dropout = hparams[HP_DROPOUT]
    batchsize = hparams[HP_BATCHSIZE]

    hparams_dict = dict(
        n_layers=n_layers,
        n_units=n_units,
        learning_rate=learning_rate,
        activation=activation,
        dropout=dropout,
        batchsize=batchsize,
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(len(features),), name="input"))

    for i in range(n_layers):
        model.add(tfpl.DenseFlipout(n_units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout, trainable=True))

    model.add(tf.keras.layers.Dense(units=2))
    model.add(tfpl.IndependentNormal(1))
    mse_scale = 10000000
    pearson_scale = 100000
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=CustomLoss(mse_scale, pearson_scale),
        metrics=[PearsonMetric(), 'mean_squared_error']
    )
    return model, hparams_dict


def main(unused_argv):
    np.random.seed(0)
    logdir = flags.FLAGS.logdir
    shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    start_clock = time.time()
    try:
        run_all(hparams, logdir=logdir, verbose=True)
        print("Done. Output saved to %s." % logdir)
    except Exception as e:
        print('error:', repr(e))

    send_to_aws(RESULTS_FILE_NAME)
    run_time = round((time.time() - start_clock) / 3600, 2)
    print(f'Search took: {run_time} hours')
    kill_ec2(run_time)


if __name__ == "__main__":
    app.run(main)
