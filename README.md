# Kilometer-Scale AI Downscaling of Atlantic Hurricanes with Generative Ensembles

Yingkai Sha, Talea L. Mayo, Ethan D. Gutmann, Lulin Xue, Andrew Newman

NSF National Center for Atmospheric Research, Boulder, Colorado, USA

Department of  Mathematics, Emory University, Atlanta, Georgia, USA

## Abstract

This study presents an AI-based dynamical downscaling system for Tropical cyclones (TCs). The system incorporates an AI-based limited-area model that downscales 3-hourly low-resolution boundary forcings into hourly high-resolution fields autoregressively, and a diffusion model that converts the outputs into ensembles of hazard-relevant variables. The system is trained on the regridded CONUS404 data with ERA5 forcings, and is evaluated on 20 TCs in 2020--2024. Verification shows stable downscaling performance across Atlantic hurricane seasons, with energy spectra closely matching the CONUS404 reference. The system is also verified to produce skillful TC-relevant weather extremes, largely improved over a deterministic AI baseline. The system performs well with forcing data from other models (GDAS/FNL) and can produce realistic eyewall, rainband, and landfall structures in TC case studies. The system offers a good example of how AI-based dynamical downscaling systems can be designed to access small-scale extreme weather events.

## Introduction

A repository based on [MILES-CREDIT](https://miles-credit.readthedocs.io/en/latest/) (`credit` package, version 2025.3.0) for the RAL GWC TC downscaling.

### Installation
* Run `create_derecho_env.sh` — creates the conda env (Python 3.11), installs the
  Derecho MPI-enabled PyTorch wheels, then `pip install -e .`.
* Alternative: MILES-CREDIT [documentation](https://miles-credit.readthedocs.io/en/latest/)

## Navigation

## Resources

## Acknowledgement

This material is based upon work supported by the National Science Foundation (NSF) National Center for Atmospheric Research (NCAR), which is a major facility sponsored by the U.S. National Science Foundation under Cooperative Agreement No. 1852977. 
Y. Sha and T. Mayo are also supported by the EdEC Faculty Innovator Program No. 1755088.
The authors acknowledge high-performance computing support from Derecho and Casper \cite{Cheyenne} provided by the Computational and Information Systems Laboratory, NCAR, and sponsored by the NSF.

