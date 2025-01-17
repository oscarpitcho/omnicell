# TODO: 
 - Will need some kind of structure to recover training in an interrupted run - Model implementations should be given the chance to load existing checkpoint
 - For now model config is managed completely by the model itself, we pass the parsed yaml file to the model
 - Random note, we will have many downstream tasks that will be separate from the model and the initial pretraining. Need a way to run these tasks in a modular way and pass the model that we want to use. We will need also to be able to define some fine tuning tasks that will be separate from the initial training not just some prediction.

 - How do we also deal with a model not just pretraining but also making predictions?
 - Saving the data such that we can modify the plot generating stuff 
 - don't name dirs as .sth --> creates issues with cluster
 - Having an instance is useful, models might want to create instances of oder models and whatnot to use the logic within them
 

 # Important 
  - VAE Code breaks if we start passing dataloader, it assumes it receives all then training 