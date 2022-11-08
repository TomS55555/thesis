# TO DO:

- [ ] Call __get_item__() on dataset from SimCLR tutorial to see what it looks like
	- Make sure own dataset does the same thing
- [ ] Find out how optional arguments work in python such that dataset augmentation can be made more efficient
- [ ] Improve train_functions
- [ ] Time the fetching of data somehow, at least make sure it is not too excessive
- [ ] Train SimCLR on a small dataset


-------------------
## MORE COMPLICATED:
- Get a better understanding of the implementation of the info_nce_loss function
- Optimize the data-loading process: find out what the bottleneck is and how the interaction between CPU
  and GPU can be optimized
 	- What is the maximum amount of patients that can be loaded in a specific amount of RAM? 
 		- How exactly does memory get allocated at the OS-level, diagnostics about this?
- Find out why opening files is so slow; what goes wrong at the OS-level?

-----------------

## Questions:
- What is the temperature parameter in nce_loss