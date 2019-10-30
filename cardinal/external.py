from alipy.query_strategy.query_labels import QueryInstanceGraphDensity


def knn_density_sampling(X: np.ndarray,
                         idx_labelled: np.ndarray,
                         n_instances: int = 1,
                         metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    
    idx_all = np.arange(X.shape[0])
    idx_unlabelled = list(set(idx_all) - set(idx_labelled))
    
    # TODO: we could allow customization of number of nearest neigbours
    sampler = QueryInstanceGraphDensity(X, None, idx_all, metric=metric)
    
    idx_next = sampler.select(idx_labelled, idx_unlabelled, batch_size=n_instances)
    
    # We want to get the confidence scores, we use the starting density in to_dict method
    confidence = sampler.to_dict()[1][idx_next]
    
    return idx_next, confidence
    