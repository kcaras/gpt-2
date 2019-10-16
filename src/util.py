def remove_min_max(arr):
    max_ele = max(arr)
    min_ele = min(arr)
    return [val for val in arr if val != max_ele and val != min_ele]
