# Vegetable Mania!!

This is a Deep Learning project that utilises TensorFlow Keras to build Convolutional Neural Networks to predict the type of vegetable present in an image. We will be using Tensorflow Serving along with Render.com to host and serve our Deep Learning model via API.

## For Developers

### Current Model Structure:

cnn_small:

```
Input (31, 31, 1)
   |
Conv2D(16, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Conv2D(32, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Conv2D(64, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Flatten
   |
Dense(64, ReLU)
   |
Dropout(0.5)
   |
Dense(15, Softmax)
```

cnn_large:

```
Input (128, 128, 1)
   |
Conv2D(32, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Conv2D(64, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Conv2D(128, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Conv2D(256, 3x3) -> BatchNormalization -> MaxPooling2D
   |
Flatten
   |
Dense(256, ReLU) -> BatchNormalization -> Dropout(0.5)
   |
Dense(128, ReLU) -> BatchNormalization -> Dropout(0.5)
   |
Dense(15, Softmax)

```


### Training Instructions

To train this model, it is highly recommended to use the dataset [hosted here](https://www.kaggle.com/datasets/moustacheman/vegetable-images). However, you have the flexibility to supply your own dataset if necessary.

If opting for the recommended dataset, download it to the root of this project with the directory name "Vegetable Images". Ensure your project structure looks like:

```
📁 Your_Project_Root
│
├── 📁 Vegetable Images
│   ├── ... (dataset contents)
│
├── main.ipynb
```

If using a different dataset, simply edit the `ROOT` constant in `main.ipynb` as required.

🚀 **CUDA-enabled GPU**: For faster training, ensure your system has a CUDA-enabled GPU.

⚙️ **Environment Setup**: Remember to configure your Python environment, preferably version 3.9.18, using the `requirements.txt` file to ensure compatibility. You can use the following command:

```bash
pip install -r requirements.txt
```

Happy training! 🌿🤭

### Deployment Instructions

Ensure your directory follows this structure:

```
📁 Your_Project_Root
│
├── 📁 cnn_small
│   ├── ... (model contents)
├── 📁 cnn_large
│   ├── ... (model contents)
|
├── Dockerfile
│
├── model_config.config
```

To test the model locally, build a Docker Container using the TensorFlow image according to the instructions given in the `main.ipynb`. 

> If you are using a M Series Macbook, use the `localDockerfile` instead for compatibility. Remember to change back the file when deploying to Render.


## Project Files

| File/Folder      | Type   | Description                                                       |
|-------------------|--------|-------------------------------------------------------------------|
| cnn_large         | Folder | Directory containing a CNN in the TensorFlow SavedModel format. Used to predict 128 x 128 images. |
| cnn_small         | Folder | Directory containing a CNN in the TensorFlow SavedModel format. Used to predict 31 x 31 images.    |
| Docs              | Folder | Contains notes I've written documenting various parts of the project. Also contained screenshots of key steps taken. |
| tests             | Folder | Running PyTest to ensure Render.com container is running.          |
| DL                | Folder | Contains notebook used to research and find optimal models.        |
| Dockerfile        | File   | Used to build the image to be deployed to Render.com.              |
| localDockerfile   | File   | For Mac users facing incompatibility with TensorFlow Serving to use and run local containers.   |
| main.ipynb        | File   | Contains notebook used to build and train the Deep Learning model. |
| model_config.cfg  | File   | Used to configure the TensorFlow model to be served.               |

