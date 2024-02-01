Compatibility Issues:
    - Can't use my final_large.keras for this project because I use a M1 mac and have custom tensorflow which is maintained by apple 
    - ca1 keras model was saved with tenforflow version 2.13.0 and I'm using 2.15.0 so some of the layers are not compatible -> got the error "Layer 'layer_normalization_2' expected 2 variables, but received 0"
    - I tried to change the tensorflow version by downgrading my mac to 2.13.0 -> got "'Adam' object has no attribute 'build'" again due to version mismatch
    - In the end I decided to retrain the model from scratch in my Mac using the old dataset
    - Tensorflow/serving doesn't work on a Mac as it is not supported yet. So I will be using a Image from this repo -> https://github.com/emacski/tensorflow-serving-arm so that it works

[old] Retraining:
    - Dataset is located in DL folder but gitignore will block it cuz its too big
    - I plan to train the model first. 
    - Since I will be using this model in "production", I will train it on the full dataset without any train-test-val splits -> Maximise the context model has
    - I realised that we need to use two different models

Handling Change in Project:
    - I have two issues right now:
        - Stop using custom CNN and use the regular Keras CNN to make it easier to deploy
        - Transfer my project to new repo as per assignment requirements
    - Now my entire repo is in a new spot and I will be starting model training and saving from scratch as I decided to change models

Next Plan:
    - I will choose two simple CNN models (31x31 and 128x128) and retrain them as I havent saved the weights during CA1
    - Save into required format and also visualse some key info for teacher to see.
    - Document Deployment process in the jupyter notebook.

Serving Process:
    - Prerequsites:
        - have a model(s) saved in a folder in the TF SavedModel format
        - have tensorflow/serving or an equivalent installed
    - Shift the Model to a different location that is more accessible -> in my case it is /Users/sriramjeyakumar
    - Run the TF Serving image using -> 
        ``` docker run --name cnn_models --restart always -p 8501:8501 -v "/Users/sriramjeyakumar/Production/model_config.config:/models/model_config.config" -v "/Users/sriramjeyakumar/Production/cnn_large:/models/cnn_large" -v "/Users/sriramjeyakumar/Production/cnn_small:/models/cnn_small" -t emacski/tensorflow-serving:latest-linux_arm64  --model_config_file=/models/model_config.config ```
        - Specify model to bind to /Users/sriramjeyakumar/Production/cnn which is where I stored my model
        - Open port 8501 to send requests to model
        - Name the model as CNN
        - Use the custom TF serving image from Emacski to be compatible with M1 Mac 

Wrong commands:
    - docker run --name cnn_models -p 8501:8501 -v /Users/sriramjeyakumar/Production/cnn/large:/models/cnn -e MODEL_NAME=large -t emacski/tensorflow-serving:latest-linux_arm64
    - docker run --name cnn_models -p 8501:8501 -v /Users/sriramjeyakumar/Production/cnn:/models/cnn -e MODEL_NAME=large -t emacski/tensorflow-serving:latest-linux_arm64
    - docker run --name cnn_models -p 8501:8501 -v /Users/sriramjeyakumar/Production/cnn/large:/models/cnn -e MODEL_NAME=large -t emacski/tensorflow-serving:latest-linux_arm64

Docker Command Fix:
    - Save models in diff directories
    - docker run --name cnn_models -p 8501:8501 -v "/Users/sriramjeyakumar/Production/cnn_large:/models/cnn_large" -e MODEL_NAME=cnn_large -t emacski/tensorflow-serving:latest-linux_arm64
    - This command works so I will modify it to load both models into one container
    - Need to use a config file to tell tensorflow to load both models.
    - docker run --name cnn_models -p 8501:8501 -v "/Users/sriramjeyakumar/Production/model_config.config:/models/model_config.config" -v "/Users/sriramjeyakumar/Production/cnn_large:/models/cnn_large" -v "/Users/sriramjeyakumar/Production/cnn_small:/models/cnn_small" -t emacski/tensorflow-serving:latest-linux_arm64  --model_config_file=/models/model_config.config

Creating Development Container:
    - Forgot that I need to do all this in a development container so it's not too late and I will be creating one
    - docker pull python:3.9.18 -> using cuz its same as my mac
    - Ran 
    ```docker run -it -v /var/run/docker.sock:/var/run/docker.sock --name CNN_Server python:3.9.18 sh -c "apt-get update ; apt-get install docker.io -y ; bash" ```
    - Used VSCode Remote Explorer to open up container and pulled my git repo
    - python -m venv en -> in the root not project folder as I dont want my env to be in the git
    - source ./env/bin/activate
    - python -m pip install tensorflow
    - pip install matplotlib seaborn numpy pytest
    - pip freeze >> ca2-daaa2b01-2214618-jeyakumarsriram-dl/requirements.txt
    - apt-get install iputils-ping
    - docker network create dl_network
    - docker network connect dl_network cnn_models
    - docker network connect dl_network CNN_Server
    - Later if needed can setup again but simply using the requirements.txt file

Testing TF Serving:
    - During the testing for the container, I found that whenever a bad input is given, the container simply crashes and stops
    - I will add a "--restart always" flag to restart contianer even if it crashes.
    - The restart policy works but


Some more fixes for creating CNN_server:
    - Realised that my docker doesn have the docker command cuz during the mounting, I only mounted the docker.sock but not the docker
    - Docker container uses 3.8 while my mac uses 3.11 so i'm going to change the python version too while I'm at it
    - Going to recreate container
    - Using 3.9.18 so I can have the same python version as my laptop for consistency
    - I finally fixed all the docker errors but having a simple script at the end to install docker.io but also preserve my docker.socket that is mounted from my computer

Testing:
    - Planned testing will include testing TF apis as well as the accuracy of the model.
    - I have written pytest to test consistency, range, expected failure and unexpected failures. 
    - For now I am testing the local URL but soon I will move to the Cloud URl once I deploy it. I will just need to change one var in the conftest.py


Deployment:
    - I used a simple dockerfile to serve my models. I just had to load my models and my config files onto the image and expose the right ports
    - While deploying my code, I faced a compatibility error. Since I have been using a custom Tensorflow/Serving image that is made for my ARM chip, the same image does not run on Render as they used Linux
    - I solved this by using the "normal" Tensorflow Serving image when deploying and a custom one when running locally. It ain't great but I have to do it
    - I also changed the URL in the pytest and reran it. As expected it works flawlessly which means my render model is working.

Changing Deployment:
    - After viewing the current state of the repo, I find that the models are very large and not suitable to be present in a normal repo. Everytime the model is changed, it si very inefficent as git is not design to track models. 
    - To fix this I will be attempting to use Git Large File Storage (LFS) to store my model isntead. 
    - To enable it, we just need to check the LFS box shown in the screen shot called "Git LFS Setting"

Setting Up Git LFS:
- For windows, it is installed by default so you dont have to
- For mac just use `brew install git-lfs`
- But since we using a linux environment inside the docker container, we will have to download the scripts and manually install it. 
- `curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash` to download. dont need sudo cuz its inside a container
- `apt-get install git-lfs` to install. It will show some errors and stuff but don't be confused, it has been installed
- Confirm installation with `git lfs` and see the output.
- Run `git lfs install` inside your project folder to initialise it in your repo
- Run `git lfs track "*/*/variables/*"` to track the files. Actually I could just track all the files inside the cnn_large and small but I realised that only the variables folder was large as it contained the weights. So I will only upload that for git lfs and keep the rest cuz they are just metadata. 


Setting up CICD:
- After trying to run a docker container within gitlab cicd and being unable to get it going, I will try a different albeit inferior approach
- I will setup another Render Container that is used to testsing. First I will deploy to that and run pytest on it. If it succeds then I will deploy to the main container
- I could do this with just one container, but I dont want to risk deploying a wrong commit to the main container before testing. That doesnt seem to be good practice. 
- I will disbale automatic redeploy in both Test and Production containers so that it would wait for my deploy hook.
- After facing some errors with the yml file, I finally managed to get the CICD working. 