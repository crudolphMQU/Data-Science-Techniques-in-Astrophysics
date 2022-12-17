# Data Science Techniques in Astrophysics

This repository contains the files for the labs for ASTR3110 in S1 2021. There are a total of 3 labs with their own individual tasks. All packages have been updated as of 15/10/21. Throughout the labs, there are references to resources used whether taught within class or outside of class.

# Lab 1 - Period Luminosity Relation Using Cepheid Variables
## Summary 
Throughout the notebook, we did some data exploration and management (filtering and cleaning) of a 'gz' VOTable file type. Once happy with the dataset, we began a sample selection for Cepheid Variables (CV) within the entire dataset and chosen classical CV specifically. These CVs are brighter and more massive which will help in our task in using the period and luminosity to determine distance and callibrating the Leavitt Law. 

We then began determining and comparing the best-fitting straight line parameterisation of the Leavitt Law using standard optimisation (reducing chisq) and Markov chain Monte Carlo (MCMC). It was also useful to compare the distances found using parallax and the distance modulus via the Wesenheit magnitude and Period Wesenheit relation which corrects for dust reddenning of the observed objects. 

Finally, we used the calibrated Leavitt Law to determine the distance to the Large Megellanic Cloud (LMC). This one done by filtering the dataset for the LMC coordinates, determine the Wesenheit magnitude and use parallax and distance modulus to compare the measured distances.

## Data
We retrieve the data from the online [Gaia mission archive](http://gea.esac.esa.int/archive/) using the following Astronomy Data Query Language (ADQL):
```
SELECT source_id, dist.r_est, dist.r_lo, dist.r_hi, dist.r_len, dist.result_flag, dist.modality_flag, src.ra, src.dec, src.L, src.B, src.parallax, src.parallax_error, cep.type_best_classification, cep.mode_best_classification, cep.pf, cep.pf_error, cep.int_average_bp, cep.int_average_bp_error, cep.int_average_g, cep.int_average_g_error, cep.int_average_rp, cep.int_average_rp_error, cep.g_absorption,cep.g_absorption_error,cep.num_clean_epochs_g
FROM external.gaiadr2_geometric_distance as dist 
JOIN gaiadr2.vari_cepheid AS cep USING (source_id) 
JOIN gaiadr2.gaia_source AS src USING (source_id)
```

This will allow us to download the 'gz' VOTable filetype once the query has completed. 

## Useful Packages
```pyrex
from astropy.io.votable import parse
from Imports import emcee

```

# Lab 2 - Using GMM to Determine Star Cluster Membership
## Summary 

## Data
Again, using  an ADQL statement to retreive the data from [Gaia mission archive](http://gea.esac.esa.int/archive/):

```
SELECT source_id, dist.r_est, dist.r_lo, dist.r_hi, dist.r_len, dist.result_flag, dist.modality_flag, src.ra, src.dec, src.L, src.B, src.parallax, src.parallax_error, src.pmra, src.pmra_error, src.pmdec, src.pmdec_error, src.radial_velocity, src.radial_velocity_error, src.astrometric_chi2_al, src.astrometric_n_good_obs_al, src.phot_g_mean_mag, src.phot_bp_mean_mag,src.phot_rp_mean_mag, src.phot_g_mean_flux, src.phot_bp_mean_flux, src.phot_rp_mean_flux, src.phot_g_mean_flux_error, src.phot_bp_mean_flux_error, src.phot_rp_mean_flux_error, src.phot_bp_rp_excess_factor 
FROM external.gaiadr2_geometric_distance as dist 
JOIN gaiadr2.gaia_source AS src USING (source_id) 
where CONTAINS(POINT('icrs', src.ra, src.dec), CIRCLE('icrs',116.3500,-37.9500,2))=1
```
## Useful Packages
```pyrex
from astropy.io.votable import parse
from sklearn.mixture import GaussianMixture as GMM
import seaborn as sns
```

# Lab 3 - Classifying Images using Artificial and Convolutional Neural Networks.
## Summary 

## Data 
The data was collected and uploaded to a private git repository for the class to download and store safely. The filesize is too large to download and can be provided upon request. The data consists of Spitzer imaging in the 4.5, 5.8, and 8.0 $\mu$m bands. There are 100 images of each object being planetary nebulae, HII regions and radio galaxies (PNE, HII, RG).

## Useful Packages
```pyrex
import glob
import itertools
from scipy import ndimage
from scipy import misc
from matplotlib.colors import LogNorm
from astropy.io import fits
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense 
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.core import Dropout
import os
```
