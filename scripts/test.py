import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.expanduser('~/Documents/Hybrid/'))
from Hybrid.verification import load_tganom_and_compute, load_tgex_and_compute, build_fit_nn_model, load, compute_bss, compute_kss, compute_auc, reduce_to_ranks

from Hybrid.dataloading import read_tganom_predictand
from Hybrid.dataprep import default_prep

#climname = 'tg-anom_clim_1998-06-07_2019-10-31_21D-roll-mean_15-t2m-q095-adapted-mean_15_15_q0.5' 
#modelclimname = 'tg-anom_45r1_1998-06-07_2019-08-31_21D-roll-mean_15-t2m-q095-adapted-mean_15_15_q0.5'
#booksname = 'books_paper3-1_tg-anom_JJA_45r1_21D-roll-mean_15-t2m-q095-adapted-mean.csv'
#df = load_tganom_and_compute(bookfile = booksname, climname = climname, modelclim = modelclimname, add_trend = True, return_trend = False)
#
#tgexname = 'books_paper3-2_tg-ex-q0.75-21D_JJA_45r1_1D_15-t2m-q095-adapted-mean.csv'
#df2 = load_tgex_and_compute(bookfile = tgexname, nday_threshold = 5, add_trend = True, return_trend = False)

total, test_only, prepared_data = build_fit_nn_model(predictandname = 'tg-anom_JJA_45r1_21D-roll-mean_q0.66_sep12-15', npreds = 4, do_climdev = False) # Sequential forward
#one, two = default_prep(predictandname = 'tg-anom_JJA_45r1_31D-roll-mean_q0.5_sep12-15', npreds = 4) # Sequential forward
#total2, test_only2 = build_fit_nn_model(predictandname = 'tg-anom_JJA_45r1_31D-roll-mean_q0.5_sep12-15', use_jmeasure = True, npreds = 6) # j_measure
#
## Possibility to add both.
#joined = test_only.copy().merge(test_only2.iloc[:,test_only2.columns.str.startswith('ppjm')], left_index = True, right_index = True)
#compute_kss(joined)
#compute_auc(joined)
#ranks = reduce_to_ranks(joined)

