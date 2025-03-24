from hpi_parego.my_local_and_random_search_configspace import MyLocalAndSortedRandomSearchConfigSpace


def test_answer():
    search = MyLocalAndSortedRandomSearchConfigSpace()
    values = {(0,):0.1, (1,):0.4, (0,1):0.4}
    res = {(0,):0.1, (1,):0.4, (0,1): 0.9}
    test = search.sum_mi_values_higher(values)
    test = {k:round(v,2) for k,v in test.items()}
    assert test == res
    
    values = {(0,):0.1, (1,):0.4, (2,):0.6, (0,1):0.4, (0,2):0.7, (1,2):0.5}
    res = {(0,):0.1, (1,):0.4, (2,):0.6, (0,1): 0.9, (0,2):1.4, (1,2):1.5, (0,1,2): 2.7}
    test = search.sum_mi_values_higher(values)
    test = {k:round(v,2) for k,v in test.items()}
    assert test == res
    
    values = {(0,):0.1, (1,):0.4, (2,):0.6, (3,):0.6, (0,1):0.4, (0,2):0.7, (1,2):0.5, (0,3): 0.1, (1,3):0.3, (2,3): 0.6}
    res = {(0,):0.1, (1,):0.4, (2,):0.6, (3,):0.6, (0,1): 0.9, (0,2):1.4, (1,2):1.5, (0,1,2): 2.7, (0,3): 0.8, (1,3): 1.3, (2,3): 1.8, (0,1,3): 1.9, (0,2,3): 2.7, (1,2,3): 3.0, (0,1,2,3): 4.3}
    test = search.sum_mi_values_higher(values)
    test = {k:round(v,2) for k,v in test.items()}
    assert test == res