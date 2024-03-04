## Testing Max Engine
* This sample project is a simple example of how to use the Max Engine to run a simple model.
* Also, time is measured to see how long it takes to run the model.

## Time Comparison
* (Mini Size) MNist model is simple CNN model and time is measured while infer all test data.
* (Medium Size) Res25 model is for loop ResNet101 model 25 times with fully-connected layer, and time is measured while infer single batch test data. 
* Both model is tested on CPU.

| Model | Load Time (s) | Run Time (s) |
|-------|---------------|--------------|
| Mnist (Torch) | 0.016 | 2.9 |
| Mnist (🔥) | 0.36 | 8.14 |
| Res25 (Torch) | 1.89 | 4.21 |
| Res25 (🔥) | 2.27 | 1.15 |


## Running the test
### Running the MNist model
* Compile Torch model into TorchScript

```bash
python mnist.scriptCompile.py
```

* Run the model with TorchScript

```bash
python mnist.run.py
```

* Run the model with Max Engine

```bash
python mnist.run-🔥.py
```

### Running the Res25 model
* Compile Torch model into TorchScript

```bash
python res25.scriptCompile.py
```

* Run the model with TorchScript

```bash
python res25.run.py
```

* Run the model with Max Engine

```bash
python res25.run-🔥.py
```
