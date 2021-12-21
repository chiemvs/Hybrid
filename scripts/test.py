import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.verification import load_tganom_and_compute, load_tgex_and_compute, build_fit_nn_model

climname = 'tg-anom_clim_1998-06-07_2019-10-31_31D-roll-mean_15-t2m-q095-adapted-mean_15_15_q0.75' 
modelclimname = 'tg-anom_45r1_1998-06-07_2019-08-31_31D-roll-mean_15-t2m-q095-adapted-mean_15_15_q0.75'
booksname = 'books_paper3-1_tg-anom_JJA_45r1_31D-roll-mean_15-t2m-q095-adapted-mean.csv'
df = load_tganom_and_compute(bookfile = booksname, climname = climname, modelclim = modelclimname, add_trend = True, return_trend = False)

tgexname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_0.01-t2m-grid-mean.csv'
df2 = load_tgex_and_compute(bookfile = tgexname, nday_threshold = 7, add_trend = True, return_trend = False)

forc_test, forc_trainval, obs_test, obs_trainval, preds_test, preds_trainval = build_fit_nn_model(predictandname = 'tg-anom_JJA_45r1_31D-roll-mean_q05_sep12-15', npreds = 4)
