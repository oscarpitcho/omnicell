# Optimizing preprocessing performance

`create_cell_graph_dataset` in `pertdata.py` is very slow and bottlenecks preprocessing (30 days expected on repogle k562). Changes made: 
- `num_samples` is always 1, this is the value given in the example notebooks. Consequently for each perturbed cell we randomly select one control cell. We replace this with just one sampling step that matches a control to each perturbed datapoint. 



- Eliminated the recurrent desparsifying steps in the final for loop and intoduced more efficient for comprehensions:
```python
cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))
```

