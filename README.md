# Traffic Light Detection with Tensorflow Object Detection API

This repository was part of the [Final Project](https://github.com/hemingchen/CarND-Capstone) I submitted to [Udacity Self-Driving Car Engineer Nano Degree Program](https://eu.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

Based on a pretrained model, it uses the Tensorflow Object Detection API to train a new model to identify traffic light state from the images captured by in-vehicle camera.


## I. Setup Tensorflow and Object Detection API

For Linux, please refer to the official installation guide:
 
<https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>

For Windows, I used the following procedures to install Tensorflow Object Detection API on Windows 10. 

Procedures mentioned in this section do not involve any file in this repository yet.

### 1. Install Windows dependencies

Download google protobuf from <https://github.com/google/protobuf/releases> and extract the contents to
```
C:\Program Files (x86)\protoc
```

`protoc.exe` should be saved to:
```
C:\Program Files (x86)\protoc\bin\protoc.exe

```

Then add `C:\Program Files (x86)\protoc\bin\` to `PATH`

### 2. Install Python dependencies
```
# To enable GPU, use tensoflow-gpu intead
pip install tensorflow pillow lxml jupyter matplotlib
```

### 3. Install Tensorflow Object Detection API source code

#### 3.1 Install the source code
```
# Create folder
mkdir C:\tensorflow
```

```
# From C:\tensorflow
git clone https://github.com/tensorflow/models.git
```

#### 3.2 Update environment variable
Create `PYTHONPATH` environment variable and add the following paths:
```
C:\tensorflow\models
C:\tensorflow\models\research
C:\tensorflow\models\research\slim
C:\tensorflow\models\research\object_detection
```

Then add `%PYTHONPATH%` to system `PATH` variable.

#### 3.3 Compile protobuf and generate Python code
```
# From C:\tensorflow\models\research
for %f in (object_detection\protos\*.proto) do protoc.exe %f --python_out=.
```

Proceed if Python files are successfully generated in `C:\tensorflow\models\research\object_detection\protos`

#### 3.4 Test installation

Now test the above installation by running the following command. It should not report any error.
```
python C:\tensorflow\models\research\object_detection\builders\model_builder_test.py
```

Run
```
# From C:\tensorflow\models\research\object_detection
jupyter notebook
```

Open and execute all cells in notebook `object_detection_tutorial.ipynb`.

It should not report any error. In the last cell, you should be able to see object detection results on test images.




## II. Clone this repository

From this section onwards, this repository will be needed.
```
git clone https://github.com/hemingchen/CarND-Capstone-Traffic-Light-Detection.git
```




## III. Prepare training data

### 1. Download training data

Labeled training/test data are kindly shared by other Udacity students, e.g. [Vatsal Srivastava](https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62) shared his data [here](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI), direct download link is [here](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing).

Download and extract the content to `REPO_ROOT\data` to get:
```
REPO_ROOT\data\real_training_data\
REPO_ROOT\data\sim_training_data\
REPO_ROOT\data\real_data.record
REPO_ROOT\data\sim_data.record
```

where the raw labeled data can be found in `real_training_data` and `sim_training data`. And the `.record` files were generated from them as required by Tensorflow Object Detection API for training. 

### 2. Generate TFRecord files for Tensorflow

For training, we can directly use the `.record` files already included in the downloaded package. Use the following command in case they need to be generated again.

#### 2.1 For simulator data
```
# From repo root
python C:\tensorflow\models\research\object_detection\dataset_tools\create_pascal_tf_record.py --data_dir=data\sim_training_data\sim_data_capture --output_path=data\sim_data.record
```

#### 2.2 For real data recorded on Carla
```
# From repo root
python C:\tensorflow\models\research\object_detection\dataset_tools\create_pascal_tf_record.py --data_dir=data\real_training_data\real_data_capture --output_path=data\real_data.record
```




## IV. Training

### 1. Download pretrained model

The SSD MobileNet model was used in this project, which can be downloaded at [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz). Extract the content to `REPO_ROOT\model` to get:
```
...
REPO_ROOT\model\saved_model\
REPO_ROOT\model\frozen_inference_graph.pb
REPO_ROOT\model\model.ckpt.index
REPO_ROOT\model\model.ckpt.meta
...
```


### 2. Train data

#### 2.1 For simulator data

Training:
```
# From repo root
python C:\tensorflow\models\research\object_detection\train.py --pipeline_config_path=config\ssd_mobilenet_v1_coco_sim.config --train_dir=data\sim_training_data\sim_data_capture
```

Save for inference:
```
# From repo root
python C:\tensorflow\models\research\object_detection\export_inference_graph.py --pipeline_config_path=config\ssd_mobilenet_v1_coco_sim.config --trained_checkpoint_prefix=data\sim_training_data\sim_data_capture\model.ckpt-6000 --output_directory=model_frozen_sim\
```


#### 2.2 For real data recorded on Carla

Training:
```
# From repo root
python C:\tensorflow\models\research\object_detection\train.py --pipeline_config_path=config\ssd_mobilenet_v1_coco_real.config --train_dir=data\real_training_data
```

Save for inference:
```
# From repo root
python C:\tensorflow\models\research\object_detection\export_inference_graph.py --pipeline_config_path=config\ssd_mobilenet_v1_coco_real.config --trained_checkpoint_prefix=data\real_training_data\real_data_capture\model.ckpt-25000 --output_directory=model_frozen_real\
```




## References

- <https://github.com/tensorflow/models>

- <https://codeburst.io/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58>

- <https://medium.com/@rohitrpatil/how-to-use-tensorflow-object-detection-api-on-windows-102ec8097699>

- <https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62>

- <https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI>

- <https://github.com/mkoehnke/CarND-Capstone-TrafficLightDetection>