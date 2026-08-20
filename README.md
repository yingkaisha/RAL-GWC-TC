# Kilometer-Scale AI Downscaling of Atlantic Hurricanes with Generative Ensembles

Yingkai Sha<sup>a</sup>, Talea L. Mayo<sup>b</sup>, Ethan D. Gutmann<sup>a</sup>, Lulin Xue<sup>a</sup>, Andrew Newman<sup>a</sup>

<sup>a</sup>NSF National Center for Atmospheric Research, Boulder, Colorado, USA

<sup>b</sup>Department of  Mathematics, Emory University, Atlanta, Georgia, USA

## Abstract

This study presents an AI-based dynamical downscaling system for Tropical cyclones (TCs). The system incorporates an AI-based limited-area model that downscales 3-hourly low-resolution boundary forcings into hourly high-resolution fields autoregressively, and a diffusion model that converts the outputs into ensembles of hazard-relevant variables. The system is trained on the regridded CONUS404 data with ERA5 forcings, and is evaluated on 20 TCs in 2020--2024. Verification shows stable downscaling performance across Atlantic hurricane seasons, with energy spectra closely matching the CONUS404 reference. The system is also verified to produce skillful TC-relevant weather extremes, largely improved over a deterministic AI baseline. The system performs well with forcing data from other models (GDAS/FNL) and can produce realistic eyewall, rainband, and landfall structures in TC case studies. The system offers a good example of how AI-based dynamical downscaling systems can be designed to access small-scale extreme weather events.

## Introduction

A repository based on [MILES-CREDIT](https://miles-credit.readthedocs.io/en/latest/) (`credit` package, version 2025.3.0) for the RAL GWC TC downscaling.

### Installation
* For NSF NCAR users: Log in to Derecho, run `create_derecho_env.sh` — creates the conda env (Python 3.11), installs the
  Derecho MPI-enabled PyTorch wheels, then `pip install -e .`.
* Other users: MILES-CREDIT [documentation](https://miles-credit.readthedocs.io/en/latest/)

## Navigation

* Model weights [[huggingface]](https://huggingface.co/yingkaisha/CONUS404-AI-TC/tree/main)
* Model architecture [[AI-based LAM]](https://github.com/yingkaisha/RAL-GWC-TC/blob/main/credit/models/swin_wrf_v2.py), [[EDM diffusion model]](https://github.com/yingkaisha/RAL-GWC-TC/blob/main/credit/models/corrdiff_unet.py)
* Downscaling domain information [[Link]](https://github.com/yingkaisha/RAL-GWC-TC/blob/main/_PAPER/visualization/FIG_Domain.ipynb)
* CRPS and energy spectrum verification [[Link]](https://github.com/yingkaisha/RAL-GWC-TC/blob/main/_PAPER/visualization/FIG_CRPS_MAE_ZES.ipynb)
* Extreme events verification [[Link]](https://github.com/yingkaisha/RAL-GWC-TC/blob/main/_PAPER/visualization/FIG_Extreme.ipynb)
* Example case Ian [[Link]](https://github.com/yingkaisha/RAL-GWC-TC/blob/main/_PAPER/visualization/FIG_example_TC_case.ipynb)

## Resources

* NSF NCAR Research Data Archive, [ERA5 Reanalysis (0.25 Degree Latitude-Longitude Grid)](https://rda.ucar.edu/datasets/d633000/)

* Google Research, Analysis-Ready, Cloud Optimized (ARCO) ERA5 [[link](https://cloud.google.com/storage/docs/public-datasets/era5)]

* Four-kilometer long-term regional hydroclimate reanalysis over the conterminous United States (CONUS404) [[link](https://gdex.ucar.edu/datasets/d559000/)]

* NCEP GDAS/FNL 0.25 Degree Global Tropospheric Analyses [[link](https://data.ucar.edu/dataset/ncep-gdas-fnl-0-25-degree-global-tropospheric-analyses-and-forecast-grids)]

* HURDAT2 [[link](https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html)]
  
## Acknowledgement

This material is based upon work supported by the National Science Foundation (NSF) National Center for Atmospheric Research (NCAR), which is a major facility sponsored by the U.S. National Science Foundation under Cooperative Agreement No. 1852977. 
Y. Sha and T. Mayo are also supported by the EdEC Faculty Innovator Program No. 1755088.
The authors acknowledge high-performance computing support from Derecho and Casper provided by the Computational and Information Systems Laboratory, NCAR, and sponsored by the NSF.

