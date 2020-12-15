# traffic_flow
Traffic flow prediction using ensemble methods.

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name --data data_name
```

You can choose "rf","lstm","gru","saes","en_1","en_2", or "en_3" as arguments for model. 
You can choose "pems" pr "nyc" as arguments for data.
The ```.h5``` weight file was saved at model folder. 
"model_pems" folder contains the trained model for pems data and "model_nyc" contains the trained model of Bike NYC data.

**Run command below to run the program:**

```
python main.py --data data_name
```
You can choose "pems" or "nyc" as arguments for data.
