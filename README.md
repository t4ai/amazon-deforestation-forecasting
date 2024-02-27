# Amazon Deforestation Forecasting Using AI/ML and IoT Data
AAI-530-Final

Final project for AAI-530: AI for IoT, University of San Diego MS-AAI.

Dashboard App: https://public.tableau.com/app/profile/tyler.foreman5220/viz/AmazonDeforestation_17087340442370/DeforestationDashboard

## Business Understanding

The Amazon Basin is covered with approximately 5.5M km2 of rainforest (Wikipedia contributors, 2024c).  Covering such a large area with traditional UAV systems would be impractical due to the need for human pilots, frequent battery swaps and lack of edge processing to handle the large volumes of data that would be collected.  Instead, a system that is fully autonomous, self-sufficient and able to operate in remote locations is required.

The proposed system couples a ruggedized UAV platform with a base station that is powered, networked and that houses robust edge compute and storage capabilities.  The base station provides docking capabilities to the UAV, with automated battery swapping and recharging, payload swaps and data offloading between missions.  The base station enables the UAV to fly nearly continuous missions throughout the day, covering large geographical areas.

The data collected is also processed on the base station, where imagery is analyzed for new deforestation activity and the results of this analysis transmitted to the cloud for use by human analysts and the forecasting models proposed as part of this project.  An onboard GPU provides sufficient edge computing capabilities to support the execution of a suite of deep learning models that can analyze the imagery and extract key insights, such as areas of deforestation activity.  The base station also provides sufficient storage to house many days worth of mission data.  As data is processed and transmitted to the cloud, it is purged from local storage.
The base station is powered by solar panels, which charge an onboard battery array.  A Starlink satellite uplink is provided to transmit data to the cloud once processed.

## Dataset

As mentioned previously, two forecasting models were developed to support deforestation monitoring and mitigation efforts.  Both models were designed to make time-series forecasts and trained on historical deforestation data collected via satellite imagery by TerraBrasilis, a data analysis infrastructure built by the Brazilian National Institute for Space Research (INPE) to support deforestation monitoring efforts.  TerraBrasilis website: http://terrabrasilis.dpi.inpe.br/en/home-page/

The target prediction variables were total deforestation area (km2) by time step and the total new deforestation activity count by time step.  The intent of this second prediction variable was to provide some context to how active a given area of deforestation was.  That is, to provide insight as to whether a deforested area is concentrated by location or dispersed across many discreet locations.  Higher activity level counts could be indicative of a spread of deforestation to new geographic areas.

## Data Preparation: Data cleanup and Feature engineering with DBSCAN clustering  -- amazon-deforestation-data-prep.ipynb

Dataset was provdied as a set of shape files containing deforestation polygons from 2008-2022 as well as descriptive layers such as areas with no naturally occuring forest and areas covered by water (hydrography).  Some light cleanup was done along with geo-merging descriptive features with predictive features.

Next, DBSCAN clustering algorithm was leveraged to geospatially cluster deforestation areas.  GeoPandas was used to calculate and add latitude and longitude features to the dataset, representing the centroid of each deforestation polygon.  After some experimentation and tuning, DBSCAN was executed with these point pairs as inputs, tuned to a 7 km radius search and a minimum cluster size of 7.  

The result was the definition of 2,223 clusters of deforestation samples.  These clusters represented geographic concentrations of observed deforestation activity.  While the clusters ranged in size and area, this provided a foundation for planning the locations of where to install base stations and provided additional granularity for the models to extract additional context from.

<img width="931" alt="image" src="https://github.com/t4ai/amazon-deforestation-forecasting/assets/132633661/ab479e9f-726f-4ed6-a9db-a1fe960b41b9">

## Model 1: Temporal Fusion Transformer (TFT) to predict deforestation area (km^2) -- amazon-deforestation-modeling-darts.ipynb

While the feature engineering work to group deforestation activity by cluster added more granularity to the time series data, the challenge of limited look-back history still remained.  However, a critical insight was made.  That is, the clusters themselves provide another dimension to the time series, effectively resulting in the creation of multiple independent time-series datasets - one for each cluster.

Based on this insight, further research was conducted to determine the best model architecture for handling time-series predictions on multiple, independent time series.  One such model architecture that is designed for this task is the Temporal Fusion Transformer (TFT).  This model, developed by Google, is designed to handle multiple independent time series and actually performs better as the number of time series reaches the thousands in count.  This model was well-suited for the engineered dataset that had been created and was selected to make the deforestation area predictions.

The TFT model was trained using Darts Framework (https://unit8co.github.io/darts/index.html).  Dataset was converted into a Darts TimeSeries dataset, grouped by Cluser ID.

## Model 2: LSTM model to predict deforestation activity counts by area  -- amazon-deforestation-modeling-activity-model.ipynb

For the second model, a more traditional approach was taken in order to compare performance against the TFT.  A Long Short Term Memory (LSTM) model architecture was selected to make the predictions for new deforestation activity count by cluster/year.  The LSTM does not inherently handle multiple time-series datasets like the one used in this project.  However, LSTM does generalize well to time series tasks in general, so it was a good foundation to start with.

The model was defined with three LSTM layers with 32, 16 and 8 units respectively.  The final LSTM fed into a dense fully connected layer that matched the size of our dataset and was coupled with a linear activation layer.
