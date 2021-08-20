import sherpa
import numpy as np 
import pandas as pd

sys.path.append('..')
from Hybrid.neuralnet import construct_modeldev_model
from Hybrid.dataprep import prepare_full_set, test_trainval_split

leadtimepool = [4,5,6,7,8] 
#leadtimepool = 13
targetname = 'books_paper3-2_tg-ex-q0.75-7D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
predictors, forc, obs = prepare_full_set(targetname, ndaythreshold = 3, leadtimepool = leadtimepool)
obs_test, obs_trainval, generator = test_trainval_split(obs, crossval = True, nfolds = 4)
X_test, X_trainval, generator = test_trainval_split(predictors, crossval = True, nfolds = 4)

# limiring X

parameters = [sherpa.Continuous(name='lr', range=[0.005, 0.1], scale='log'),
              sherpa.Continuous(name='dropout', range=[0., 0.4]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
              sherpa.Discrete(name='num_hidden_units', range=[100, 300]),
              sherpa.Choice(name='activation', range=['relu', 'elu', 'prelu'])]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=150)
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False)

for trial in study:
    model = init_model(trial.parameters)
    for trainind, valind in generator:
        training_error = model.fit(epochs=1)
        validation_error = model.evaluate()
        study.add_observation(trial=trial,
                              iteration=iteration,
                              objective=validation_error,
                              context={'training_error': training_error})
    generator.reset()
    study.finalize(trial)
