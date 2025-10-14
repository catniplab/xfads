import torch
import functools


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
