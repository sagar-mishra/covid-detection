# covid-detection

Covid detection using CNN deep learning architecture, here we are considering two classes i.e Covid and Normal therefore it is a binary image classification problem.

Data Sources : <br/>
Covid data : https://github.com/ieee8023/covid-chestxray-dataset <br/>
Normal data : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia <br/>

<h3> To Run this project </h3> 

* Pull the docker image from docker hub : <br/>
docker pull mishrasagar25/covid-detection-api:latest

* Then use docker run command to create a container : <br/>
docker run -p 5000:5000 -ti mishrasagar25/covid-detection-api /bin/bash

* Now open postman to hit flask API:
Use this end point for covid prediction : <br/>
http://localhost:5000/predict

NOTE : Use POST method and set content-type to multipart/form-data in header and in body select form-data and in key enter **image** and set type as file and then select file.

<h2> Sample Cases </h2>

* Covid Detected patient Case : 
![Covid example](resources/images/covid_example.PNG)

* Normal patient Case : 
![Normal example](resources/images/normal_example.PNG)



