# TO DO:

- [x] Implement data augmentations
- [x] Implement SimCLR and make sure it trains
- [x] Implement a classifier on top of SimCLR
- [x] Train SimCLR on cluster
- [ ] Improve efficiency of data augmentations
- [x] Maybe look at the clustering of data points in the high dimensional space with e.g. t-SNE

- [ ] Implement contrastive transformations at a batch level such that they can be done on the GPU because CPU is bottleneck
- [ ] Implement other way to load dataset (use h5py fileformat) and compare its speed in a jupyter notebook to the other 
- [ ] Connect to google cloud and try to train a simple network there with a simple setup
- [ ] Hire better equipment from google cloud and train SimCLR on whole dataset
- [ ] Load different dataset every epoch

-------------------
### More complicated:
- Get a better understanding of the implementation of the info_nce_loss function
- Optimize the data-loading process: find out what the bottleneck is and how the interaction between CPU
  and GPU can be optimized
 	- What is the maximum amount of patients that can be loaded in a specific amount of RAM? 
 		- How exactly does memory get allocated at the OS-level, diagnostics about this?
    - https://towardsdatascience.com/efficient-pytorch-part-1-fe40ed5db76c 
- Find out why opening files is so slow; what goes wrong at the OS-level?

-----------------

## Questions:
- What labels correspond to which sleep stage?
- What are the important features of EEG-signals that make classification possible? How do humans do it?
  - The algorithm should be optimized for this!