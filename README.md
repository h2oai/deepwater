# Deep Water

### What it is
* Native implementation of Deep Learning models for GPU-optimized backends (MXNet, Caffe, TensorFlow, etc.)
* State-of-the-art Deep Learning models trained from the H2O Platform
* Train user-defined or pre-defined deeplearning models for image/text/H2OFrame classification from Flow, R, Python, Java, Scala or REST API
* Behaves just like any other H2O model (Flow, cross-validation, early stopping, hyper-parameter search, etc.)
* The next best thing after sliced bread
* Under development

### What it is not
* An oil drilling platform

### Roadmap, Architecture and Demo
Download the [Deep Water overview slides](./architecture/deepwater_overview.pdf).

![](./architecture/deepwater_overview/deepwater_overview.001.jpeg "Deep Water Roadmap")
![architecture](./architecture/deepwater_overview/deepwater_overview.002.jpeg "More Data")
![architecture](./architecture/deepwater_overview/deepwater_overview.003.jpeg "Deep Water Networks")
![architecture](./architecture/deepwater_overview/deepwater_overview.004.jpeg "Deep Water Architecture")
![architecture](./architecture/deepwater_overview/deepwater_overview.005.jpeg "Deep Water Example in Flow")

### DIY Build Instructions:
If you want to use Deep Water in H2O-3, you'll need to have a .jar file that includes backend support for at least one of MXNet, Caffe or TensorFlow.

#### 1. Build MXNet 
[Instructions to build MXNet](https://github.com/h2oai/deepwater/tree/master/mxnet)

#### 2. Build TensorFlow 
[Instructions to build TensorFlow](https://github.com/h2oai/deepwater/tree/master/tensorflow)

#### 3. Build Caffe 
Coming soon.

#### 4. Build H2O Backend Connectors
From the top-level of the deepwater repository, do
```
DEEPWATER=1 ./gradlew build -x test
```

This will create the following file: `build/libs/deepwater-1.0-SNAPSHOT-all.jar`

#### 5. Add DeepWater support to H2O-3
You need to check out the [deepwater branch of h2o-3](http://github.com/h2oai/h2o-3/tree/deepwater/).
Copy the freshly created jar file `build/libs/deepwater-1.0-SNAPSHOT-all.jar` from the previous step to H2O-3's library `h2o-3/lib/deepwater-1.0-SNAPSHOT-all.jar` and you're done!

##### Build H2O-3 as usual:
```
./gradlew build -x test
```

This H2O version will now have GPU Deep Learning support!

Alternatively, for the following system dependencies, we provide recent builds for your convenience. 

* Ubuntu 16.04 LTS
* Latest NVIDIA Display driver
* CUDA 8 (latest available) in /usr/local/cuda
* CUDNN 5 (inside of lib and include directories in /usr/local/cuda/)

In the future, we'll have more pre-built jars for more OS/CUDA combinations.


##### Install the Python wheel:
```
sudo pip install h2o-3/h2o-py/h2o-3.11.0.99999-py2.py3-none-any.whl
```
If you didn't build it yourself, download a recent version of the [h2o python wheel](https://slack-files.com/T0329MHH6-F2M4B2ZFW-d77a43ebbf)

##### (Optional) Install the Python egg for MXNet
If you want to build your own MXNet models (from Python so far), install the MXNet wheel (which was built together with MXNet above):
```
sudo easy_install deepwater/thirdparty/mxnet/python/dist/mxnet-0.7.0-py2.7.egg
```
If you didn't build it yourself, download a recent version of the [mxnet python egg](https://slack-files.com/T0329MHH6-F2M2XU01H-558dd94fcc)


### Running GPU enabled Deep Water in H2O
#### (Optional) Launch H2O by hand and build Deep Water models from Flow (`localhost:54321`)

```
java -jar build/h2o.jar
```
If you didn't build it yourself, download a recent version of [h2o.jar](https://slack-files.com/T0329MHH6-F2M4AHKS8-59ac335243)

#### Java example use cases
Example [Java GPU-enabled unit tests](https://github.com/h2oai/h2o-3/tree/deepwater/h2o-algos/src/test/java/hex/deepwater).

#### Python example use cases
Example [Python GPU-enabled unit tests](https://github.com/h2oai/h2o-3/tree/deepwater/h2o-py/tests/testdir_algos/deepwater).

#### R example use cases
Coming soon.

#### Scala / Sparkling Water example use cases
Coming soon.
