---
# **Machine Learning for Vision and Multimedia (01URPOV)**
University project for the course 01URPOV - Machine Learning for Vision and Multimedia - at Politecnico di Torino.

---

### Authors:
*   Bramucci Roberto (s303683)
*   Paoli Leonardi Francesco (s297078)
<br/>

The project focuses on the **_identification_** of the **_dynamical model_** of a **_physical system_**, in order to **_control_** it (by means of a suitable control law) and **_forecast its future behavior_** over time.

This is a fundamental problem in the **_automatic control and estimation_** field, which is tackled here by exploiting Deep Learning (DL) models, properly designed, with specific focus on **_recurrent and convolutional approaches_**.

The physical system we considered is the gimbal's yaw motion (i.e., gimbal motion around the $z$ vertical axis) of a two-wheeled self-balancing mobile robot, which is showed below.


![br_video](pictures/balancing_robot/balancing_robot_movements.gif)
<br/>
<br/>

---
## User instructions
To execute the project's code:
1. Download the `notebook.ipynb`, located in the project's root folder.
2. Load and open it on Google Colaboratory.
3. In the top menu, press _Runtime > Run all_.

To modify the behaviour of the code, further indications have been provided in the notebook's comments.
<br/>
<br/>

---
## Models architectures
![models_arch](pictures/models_architecture/models_arch.png)
<br/>
<br/>

---
## Loss, metrics and performance of trained models
### Model 01
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot01_delta_metrics](./pictures/train_valid_metrics/01_delta.png)
Example of prediction on **_delta_** test set:
![plot01_delta_pred](pictures/prediction_performance/01_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot01_contig_metrics](pictures/train_valid_metrics/01_contig.png)
Example of prediction on **_contig_** test set:
![plot01_contig_pred](pictures/prediction_performance/01_contig.png)
<br/>
<br/>

### Model 02
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot02_delta_metrics](pictures/train_valid_metrics/02_delta.png)
Example of prediction on **_delta_** test set:
![plot02_delta_pred](pictures/prediction_performance/02_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot02_contig_metrics](pictures/train_valid_metrics/02_contig.png)
Example of prediction on **_contig_** test set:
![plot02_contig_pred](pictures/prediction_performance/02_contig.png)
<br/>
<br/>

### Model 03
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot03_delta_metrics](pictures/train_valid_metrics/03_delta.png)
Example of prediction on **_delta_** test set:
![plot03_delta_pred](pictures/prediction_performance/03_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot03_contig_metrics](pictures/train_valid_metrics/03_contig.png)
Example of prediction on **_contig_** test set:
![plot03_contig_pred](pictures/prediction_performance/03_contig.png)
<br/>
<br/>

### Model 04
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot04_delta_metrics](pictures/train_valid_metrics/04_delta.png)
Example of prediction on **_delta_** test set:
![plot04_delta_pred](pictures/prediction_performance/04_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot04_contig_metrics](pictures/train_valid_metrics/04_contig.png)
Example of prediction on **_contig_** test set:
![plot04_contig_pred](pictures/prediction_performance/04_contig.png)
<br/>
<br/>

### Model 05
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot05_delta_metrics](pictures/train_valid_metrics/05_delta.png)
Example of prediction on **_delta_** test set:
![plot05_delta_pred](pictures/prediction_performance/05_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot05_contig_metrics](pictures/train_valid_metrics/05_contig.png)
Example of prediction on **_contig_** test set:
![plot05_contig_pred](pictures/prediction_performance/05_contig.png)
<br/>
<br/>

### Model 06
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot06_delta_metrics](pictures/train_valid_metrics/06_delta.png)
Example of prediction on **_delta_** test set:
![plot06_delta_pred](pictures/prediction_performance/06_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot06_contig_metrics](pictures/train_valid_metrics/06_contig.png)
Example of prediction on **_contig_** test set:
![plot06_contig_pred](pictures/prediction_performance/06_contig.png)
<br/>
<br/>

### Model 07
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot07_delta_metrics](pictures/train_valid_metrics/07_delta.png)
Example of prediction on **_delta_** test set:
![plot07_delta_pred](pictures/prediction_performance/07_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot07_contig_metrics](pictures/train_valid_metrics/07_contig.png)
Example of prediction on **_contig_** test set:
![plot07_contig_pred](pictures/prediction_performance/07_contig.png)
<br/>
<br/>

### Model 08
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot08_delta_metrics](pictures/train_valid_metrics/08_delta.png)
Example of prediction on **_delta_** test set:
![plot08_delta_pred](pictures/prediction_performance/08_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot08_contig_metrics](pictures/train_valid_metrics/08_contig.png)
Example of prediction on **_contig_** test set:
![plot08_contig_pred](pictures/prediction_performance/08_contig.png)
<br/>
<br/>

### Model 09
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot09_delta_metrics](pictures/train_valid_metrics/09_delta.png)
Example of prediction on **_delta_** test set:
![plot09_delta_pred](pictures/prediction_performance/09_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot09_contig_metrics](pictures/train_valid_metrics/09_contig.png)
Example of prediction on **_contig_** test set:
![plot09_contig_pred](pictures/prediction_performance/09_contig.png)
<br/>
<br/>

### Model 10
MSE (Loss), MAE and $R^{2}$ metrics on **_delta_** train/validation data:
![plot10_delta_metrics](pictures/train_valid_metrics/10_delta.png)
Example of prediction on **_delta_** test set:
![plot10_delta_pred](pictures/prediction_performance/10_delta.png)
MSE (Loss), MAE and $R^{2}$ metrics on **_contig_** train/validation data:
![plot10_contig_metrics](pictures/train_valid_metrics/10_contig.png)
Example of prediction on **_contig_** test set:
![plot10_contig_pred](pictures/prediction_performance/10_contig.png)
<br/>
<br/>


