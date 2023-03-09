### This is a Practice/Experimentation Repository for the purpose of learning and understanding the concepts of transfer learning.
### I made this purely for my own learning and experimentation purposes. I am not a professional in this field and I am not an expert in this field. I am just a beginner who is trying to learn and understand the concepts of transfer learning and fine tuning in the context of Deep Learning.
-----
# Models used in this repository are :
- EfficientNetB0-b6
- ResNet50
- YOLOv5
# Installation from [TensorFlow - pip ](https://www.tensorflow.org/install/pip)
### Recommended python version is 3.6-3.9
> :warning:Warning: TensorFlow 2.5.0 requires cudatoolkit=11.0, but cudatoolkit=11.2 is installed. This may cause compatibility issues.
```console
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

```

```console
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"
# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
