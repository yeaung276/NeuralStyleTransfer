## Neural Style Transfer

### Introduction

This is Neural Style Transfer implementation using vgg-net-deep-19 conv-net. The hypermeters in here are used from Coursera-coursed(Convolutional Neural Network). You can expreiment with different parameter in [this colab notebook](https://colab.research.google.com/github/yeaung276/NeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb).

### How to run the project

You need to have docker installed in your machine. Run
`docker-compose up`
to start the service and go to
`localhost:8000` to interact with the app. For manual running `core.NST`, `main.py` file is provided. You will need to download VGG weight metrix from [here](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat).
<br>
useful links:
- `localhost:8000` for interacting with app
- `localhost:8001` for interacting with redis 
- `localhost:8000/docs` for interactive api documentation

### Project Architecture
This app use 
- `redis` for saving app data and outputs,
- `fastApi` for inferencit ML model
- `websocket` for realtime notification
- `asyncio` and `multiprocessing` for running the NST job in the background, and
- `docker` for containerization the app

<figure>
  <img src="https://github.com/yeaung276/NeuralStyleTransfer/blob/master/docs/app_architecture.png?raw=True" alt="architecture-diagram"/>
  <figcaption>Architecture Diagram</figcaption>
</figure>

<figure>
  <img src="https://github.com/yeaung276/NeuralStyleTransfer/blob/master/docs/app_frontend.png?raw=True"/>
  <figcaption>Frontend</figcaption>
</figure>

In this app, if content image and style image are not same size, procrocessor will center crop the images
Enjoy :D<br>

Here are some image I tried.<br>
![akh_resize](https://user-images.githubusercontent.com/58524393/89060005-35368580-d388-11ea-9572-a44edcb75cd0.jpg)
![monelisa_resize](https://user-images.githubusercontent.com/58524393/89060014-3798df80-d388-11ea-963d-3b8e22fbf4da.jpg)<br>
![akh](https://user-images.githubusercontent.com/58524393/89059967-29e35a00-d388-11ea-969e-9d4d2dc5d5a9.jpg)<br>
###### A person Image in Monalisa style <br>

![aShipinOcean_resize](https://user-images.githubusercontent.com/58524393/89060222-ae35dd00-d388-11ea-854e-3bacd16257af.jpeg)
![starryNight_resize](https://user-images.githubusercontent.com/58524393/89060232-b130cd80-d388-11ea-8265-c9d0d83d6c79.jpg)<br>
![aShipinOcean](https://user-images.githubusercontent.com/58524393/89060258-ba219f00-d388-11ea-9ace-25da9e5ee9ec.jpg)<br>
###### A ship in Ocean Image in starryNight style<br>

![louvre_small](https://user-images.githubusercontent.com/58524393/89060494-22708080-d389-11ea-91c7-57ee51314a4f.jpg)
![monet](https://user-images.githubusercontent.com/58524393/89060502-24d2da80-d389-11ea-9fb8-af1e3a4fbc59.jpg)
![louvre](https://user-images.githubusercontent.com/58524393/89060512-27cdcb00-d389-11ea-8940-3756c7b0d30e.jpg)
###### louvre building in monet painting style

`#NST` `#tensorflow` `#docker` `#redis` `#fastapi` `#websocket` `#CNN` `#transferlearning` `#python-multiprocessing`

