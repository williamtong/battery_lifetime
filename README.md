# Battery Lifetime Prediction Model

<figure>
    <img src='./images/battery-charged-and-uncharged.png' width="500">
    <figcaption>Figure 1: Courtesy: Issac's science biog</figcaption>
</figure>


## Acknowledgments
I want to thank Dr. Lawrence Pan for helping me understand the intracacies of battery chemistry, and acknowledge the authors of the two <i>Nature</i> papers (See <i>Data</i> section below) for performing the experiments and sharing their data with the world.


## Goal
This project aims to develop a predictive model for estimating battery lifetime using machine learning techniques.  It is employs only the initial 100 cycles to estimate how many cycles it would take for a battery’s state of health [SOH](https://en.wikipedia.org/wiki/State_of_health) to drop to 90%.

## Executive Summary
1. We built a predictive Random Forest Model (RFM) to estimate the number of cycles required for a battery’s SOH to drop to 90%, based on the first 100 cycles of data.
2. Our model has a median absolute percentage error (MdAPE) of 8.02% and a coefficient of determination (R²) of 0.653.

#### Model Performance Summary
| Model | Description | Holdout R² | Holdout MdAPE | Holdout RMSE |
| --- | --- | --- | --- | --- |
| GBM | 4 sets of cyclic features (No PSD features) | 0.478 | 16.6% | 234 |
| GBM | 4 sets of cyclic features + PSD features | 0.723 | 12.2% | 171|
| RFM | 4 sets of cyclic features (No PSD features) | 0.598 | 8.86% | 206 |
| RFM | 4 sets of cyclic features + PSD features | 0.653 | <u>**8.02%**</u> | 191 |

Table 1: Summary of performances of the four models.  


# Introduction

## Data
The data was obtained from the following source: 
[link](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204).  The experiment was summarized in [Severson et al.](https://www.nature.com/articles/s41560-019-0356-8) and in [Attila et al.](https://www.nature.com/articles/s41586-020-1994-5)  We initially used the following features for each cycle from 2 to 102:

1. Charge Capacity (in mAh).
2. Mean Temperature (T).
3. Maximum Internal Resistance.
4. Cycle Duration (Time).

Per the authors, the batteries were sourced from the [A123Systems](./images/battery_spec_sheet.pdf). Each battery received a different charging protocol (waveform). We featurized this by converting the charge/discharge current (CC) or voltage (CV) to their respective Power Spectral Density (PSD). 

## Models
Due to the small size of our dataset (132 batteries total), using deep learning or neural network models was not practical. Instead, we focused on two decision-tree-based models: **Random Forest Model (RFM)** and **Gradient Boosting Model (GBM)**. 

There many metrics to evaluate the efficacy of a model.  The one we choose is MdAPE (Median Average Percent) because
it is more robust to outliers.
1.  It uses <i>median</i> instead of mean.
2.  Unlike RMSE, it does not involve squaring of errors, which tends to exaggerate outlier errors.

We see the RFM model performing better than the GBM model, which typically performs better.  This is likely because of the small size of the data set.  Below show the actuall vs. the predicted <i>Training</i> data set for the two model.  One can see a clear sign of <i>overfitting</i> in the GBM model, but not in the RFM model.
<figure>
    <img src='./images/GBM_Train.png' width="500">
    <img src='./images/RFM_Train.png' width="500">
    <figcaption>Figure 2: Scatter plots of the predicted vs holdout data of the GBM (Top) and RFM (Bottom) model.  The apparent perfect fit of the GBM model is a clear sign of overfitting.</figcaption>
</figure>

The tailing off of both curves at high value is also a sign of the small data set size, since there are not enough data points at the extremities to "pull" the predictions up from the average.

## Shapley Analysis
Shapley analysis revealed that the most impactful features were the PSD areas of the higher harmonics, confirming the hypothesis that the charging protocol significantly influences battery SOH.

# Background

### Data Source and Bias Considerations
The dataset originates from Severson et al., "Data-driven prediction of battery cycle life before capacity degradation," *Nature Energy*, Volume 4, pages 383–391 (2019). The study employed a Bayesian Reinforcement Learning (BRL) method to optimize the charging protocol.

- The dataset may contain bias due to how the BRL method explores the independent variable space.
- Despite this, it includes sufficient variation to be useful for training predictive models.

Typical charging/discharging protocols are shown below. Some protocols exhibit a periodicity of 60 minutes, while others are closer to 50 minutes.

<figure>
    <img src='./images/four_ex_charge_protocol.png' width="800">
    <figcaption>Figure 3: Example charging/discharging protocols</figcaption>
</figure>


Each battery underwent approximately 90–1500 cycles of testing, with experiments often terminating soon after SOH dropped below 90%. Measurement intervals varied, as shown in the histogram below.  Despite the large variations in the measurement time intervals, the PSDs still emerge to be the most pwoerful features in the model.  This is a testament to the importance of the charging protocol to the lifetime of the batteries.

<figure>
    <img src='./images/time_intervals.png' width="1000">
    <figcaption>Figure 4: Histogram of time intervals between measurements</figcaption>
</figure>

### Internal Resistance
Longer-lifetime batteries exhibit lower internal resistance. Below is the internal resistance evolution over the testing period.


<figure>
    <img src='./images/internal_resistance.png' width="1000">
    <figcaption>Figure 5: Internal resistance evolution</figcaption>
</figure>


### Temperature
Temperature variations were small, though minor fluctuations occurred during the first 100 cycles.

<figure>
    <img src='./images/temperature.png' width="1000">
    <figcaption>Figure 6: Temperature evolution</figcaption>
</figure>


### Cycle Time
There were two main cycle durations (~48 and ~55 minutes). Batteries with shorter cycles tended to have longer lifetimes.

<figure>
    <img src='./images/cycle_time.png' width="1000">
    <figcaption>Figure 7: Cycle time analysis</figcaption>
</figure>


### Charge Capacity
Batteries with longer lifetimes exhibited a smaller initial surge in charge capacity during the first 100 cycles.

<figure>
    <img src='./images/charge_capacity.png' width="1000">
    <figcaption>Figure 8: Charge capacity evolution</figcaption>
</figure>


### Power Spectral Density (PSD) of the charge/discharge current and voltage
The most important features derived from this were:
1. Frequency of the first harmonic (fundamental).
2. Peak areas of the first 10 harmonics.

<figure>
    <img src='./images/PSD_CC.png' width="1000">
    <img src='./images/PSD_CV.png' width="1000">
    <figcaption>Figure 9: PSD of the current (top) and voltage (bottom) waveform</figcaption>
</figure>


### Metrics
We listed three metrics in Table 1 for each model.  We believe MdAPE is the most appropriate metric because it targets a criterion that is more realistic to one in our likely use case, which is tha actually lifetime of the battery (R² does not).  It is also less sensitive to outliers than RMSE.


## Modeling
We implemented the RFM model using the scikit-learn library and the GBM model using XGBoost. In general:

- The Random Forest Model builds deep trees that can overfit, but decorrelating multiple trees helps mitigate this.
- The Gradient Boosting Model builds shallow trees sequentially, reducing residual errors but increasing computational time.
- Despite GBM's reputation for superior performance, **RFM** performed better in our case due to the small dataset.

<figure>
    <img src='./images/GBM_holdout_scatterplot.png' width="500">
    <img src='./images/RFM_holdout_scatterplot.png' width="500">
    <figcaption>Figure 10: Scatter plots of the predicted vs holdout data of the GBM (Top) and RFM (Bottom) model</figcaption>
</figure>


## Shapley Values: Feature Importance
Shapley values assess the contribution of each feature to the model’s predictions. They provide local, consistent explanations of model behavior. We computed Shapley values using the Python SHAP library.

- The most important features are the high harmonic features of the charge waveforms.  The red features being on the right indicates the smaller the higher harmonics, the longer the lifetime.  This is consistent with the findings of the source of the data [link](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204), which stated the optimal charging protocols tended to be "similar to single-step constant-current charging," which would natural lack the higher harmonic features of multi-step charging protocols.

- The second most important class of features are Cycle Time, which is how long a charge cycle last.  The red features being on the right indicates shorter cycle time would lead to longer life.  This is counter-intuitive, as the standard understanding is slower charging would lead to longer lifetime.  The authors "speculate that minimizing parasitic reactions caused by heat generation may be the operative optimization strategy for these cells, as opposed to minimizing the propensity for lithium plating."  We also believe that it is possible the shorter cycles tend to have simpler waveforms, which leads to longer lifetime.

The top 30 most predictive features are shown below.

<figure>
    <img src='./images/shapley/shapley_Top30.png' width="500">
    <figcaption>Figure 11: Shapley values for the top 30 features</figcaption>
</figure>

Other SHAP plots can be found in this [folder](./images/shapley/).


## Summary and Conclusions
1. Our *best model* has a MdAPE of **8.02%** and a coefficient of determination (R²) of **0.653**.
2. **Including PSD features of the charge/discharge waveform significantly improved model performance.** This aligns with the hypothesis that charging protocols impact battery SOH.  
3. **The RFM model outperformed GBM in this dataset.** This was especially evident at extreme values of the predicted variable.
4. **We prioritized MdAPE over RMSE due to the limited dataset size.** The best-performing model was RFM with PSD features.
