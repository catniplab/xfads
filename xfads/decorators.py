import torch
import functools
import gc


def memory_cleanup(func):
    """
    A funcrion decorator ot manually Collect and clear garbage and cache before and/or after executing the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Collect and clear garbage and cache before executing the function.
        # gc.collect()
        # gc.garbage.clear()
        # torch.cuda.empty_cache()

        result = func(*args, **kwargs)

        # Collect and clear garbage and cache after executing the function.
        gc.collect()
        gc.garbage.clear()
        torch.cuda.empty_cache()

        return result

    return wrapper


def show_memory_stats(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            print(
                f"\nBefore {func.__name__}\nAllocated: {torch.cuda.memory_allocated(0) / 1e9} GB   Reserved: {torch.cuda.memory_reserved(0) / 1e9} GB   Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9} GB\n"
            )
            result = func(*args, **kwargs)
            print(
                f"\nAfter {func.__name__}\nAllocated: {torch.cuda.memory_allocated(0) / 1e9} GB   Reserved: {torch.cuda.memory_reserved(0) / 1e9} GB   Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9} GB\n"
            )
        else:
            print(
                "You are using the show_memory_stats function decorator but, no CUDA detected, function executed casually."
            )
            result = func(*args, **kwargs)

        return result

    return wrapper


def apply_memory_cleanup(cls):
    """
    A class decorator to apply the memory_cleanup decorator to all callable attributes, i.e. methods, in the class.
    """
    for attr_name, attr_value in cls.__dict__.items():
        # attr = getattr(cls, attr_name)

        if callable(attr_value):
            setattr(cls, attr_name, memory_cleanup(attr_value))

    return cls
