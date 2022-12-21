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
Using position (RA, DEC), proper motion (pmRA, pmDEC) and distance measurements from GAIA to identify open star clusters with Gaussian Mixture Modelling (GMM). By checking the clustered stars, we can Identify which stars are most likely gravitationally bound members of the cluster. 

We start off with data management (filtering and cleaning) followed by some visualisation to better understand the data we're working with, i.e. plotting the proper motion for the first cluster found by eye, we can see there are a large number of stars within the cluster that have similar proper motions. We conduct a filter such that the ratio of the fluxes for each colour band to it's error is greater than 10, this will help in reducing the number of noisy data points. 

We use GMM to identify clusters of stars within the dataset, where we previously did this by eye using various plotting packages. We start experimenting with GMM using only 2 gaussian components (RA DEC) to fit the data and then move on to 4 components (RA, DEC, pmRA, pmDEC). After some experimentation, we were able to clearly show a tight cluster within the entire dataset. We also attempted to use a Bayesian Information Criteria (BIC) to determine the optimal number of components to use in GMM. 

Lastly, we plot a colour-magnitude diagram of the different clusters found to visualise the cluster members evolution stage. This will provide insight to the colour of the star and it's brightness. 

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
Within the dataset we have 3 astrophysical objects being planetary nebulae, HII regions and radio galaxies (PNE, HII, RG) from the [CORNISH](https://cornish.leeds.ac.uk/public/index.php) survey to search for massive star formation regions. These regions ionise surrounding gas and emit radio wavelengths and can be detected as PNE and RG objects.

In the first part of the lab we use Artificial Neural Networks (ANN) to classify the images using training and testing datasets. We found that HII and RG were more easily classified than PNE due to the complexity of PNE objects. We needed to prepare the data for the ANN which involved resizing, normalising and flattening the image. This is done as the ANN requires flattened 1D images and the data preparation does just this. We tweak the hyperparameters and evaluate the performance of the model afterwards. The hyperparameters that can be tweaked for ANN are the learning rate, momentum, number of epochs, number of extra layers and number of neurons in each layer. 

In the second part of the lab we use Convolutional Neural Networks (CNN) to classify the images again to compare the performance to the ANN. The data preparation for the CNN model is similar to ANN except there are now 3 'colors' for the Spitzer telescope observation bands. So in this case, there is just one extra dimension compared to the ANN data structure. The number of hyperparameters that can be tweaked is the same as ANN, though we added the capability to have the same number of neurons or descending number of neurons for each extra layer added. 

Lastly, we performed some data augmentation to improve the CNN model. The first run of the tuned CNN model with the augmented data has significantly imrpoved results. The first run had an accuracy of 0.85 and minimised the loss function to 0.4. 

## Data 
The data was collected and uploaded to a private git repository for the class to download and store safely. The filesize is too large to download and can be provided upon request. The data consists of Spitzer imaging in the 4.5, 5.8, and 8.0 $\mu$m bands from the [CORNISH](https://cornish.leeds.ac.uk/public/index.php) survey. There are 100 images of each object (PNE, HII, RG).

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
