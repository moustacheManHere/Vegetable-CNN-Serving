Compatibility Issues:
    - Can't use my final_large.keras for this project because I use a M1 mac and have custom tensorflow which is maintained by apple 
    - ca1 keras model was saved with tenforflow version 2.13.0 and I'm using 2.15.0 so some of the layers are not compatible -> got the error "Layer 'layer_normalization_2' expected 2 variables, but received 0"
    - I tried to change the tensorflow version by downgrading my mac to 2.13.0 -> got "'Adam' object has no attribute 'build'" again due to version mismatch
    - In the end I decided to retrain the model from scratch in my Mac using the old dataset

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