# Software architecture

- data/ 
  - Contains the raw data
- trained_models/
  - Contains the checkpoints of succesfully trained models
- checkpoints/
  - Checkpoints of ongoing learning/testing
- models/
  - Pytorch definition of different models
- datasets/
  - Pytorch wrapper for dataset in the form of a datamodule
- trainers/
  - Pytorch-lightning implementation of the trainers for the different models
- utils/
  - Some utility functions
- experiments/
  - The json files for the training of the different models
- results/
  - Jupyter notebooks showcasing the results