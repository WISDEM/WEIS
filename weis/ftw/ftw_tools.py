def split_list_chunks(fulllist, max_n_chunk=1, item_count=None):
    # Split items in a list into nested multiple (max_n_chunk) lists in an outer list
    # This method is useful to divide jobs for parallel surrogate model training as the number
    # of surrogate models trained are generally not equal to the number of parallel cores used.

    item_count = item_count or len(fulllist)
    n_chunks = min(item_count, max_n_chunk)
    fulllist = iter(fulllist)
    floor = item_count // n_chunks
    ceiling = floor + 1
    stepdown = item_count % n_chunks
    for x_i in range(n_chunks):
        length = ceiling if x_i < stepdown else floor
        yield [next(fulllist) for _ in range(length)]
