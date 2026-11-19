# Predictive Maintenance & Anomaly Detection for Solar PV & Industrial Inverters

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Anomaly%20Detection-orange?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Failure%20Prediction-green)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-red)

## Project Overview
Unplanned inverter downtime in utility-scale solar and industrial power distribution leads to significant energy yield losses. This end-to-end machine learning pipeline combines **domain-specific electrical & thermal feature engineering**, **unsupervised Isolation Forest anomaly scoring**, **class-balanced LightGBM fault prediction**, and **SHAP root-cause explainability** to predict hardware failures up to 7 days in advance.

---

##  System Architecture

```text
+-----------------------+      +----------------------------------+      +---------------------------+
|  Inverter Sensor      | ---> |  Electrical & Thermal Features   | ---> |  Hybrid ML Modeling       |
|  Telemetry (4,000)    |      |  • IGBT Thermal Gradient         |      |  • Isolation Forest Score |
+-----------------------+      |  • Grid Frequency Deviation      |      |  • Balanced LightGBM      |
                               |  • THD & Voltage Stress Index    |      +-------------+-------------+
                               +----------------------------------+                    |
                                                                                       v
+-----------------------+      +----------------------------------+      +---------------------------+
|  Priority Maintenance | <--- |  SHAP Root-Cause Diagnostics     | <--- |  Imbalanced Evaluation    |
|  Alert Schedule (CSV) |      |  (Component Stress Attribution)  |      |  (ROC-AUC & PR-AUC)       |
+-----------------------+      +----------------------------------+      +---------------------------+
