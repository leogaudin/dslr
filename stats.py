
def ft_mean(args: list[int | float]):
    """Returns the mean of the provided list.
    """
    return sum(args) / len(args)


def ft_median(args: list[int | float]):
    """Returns the median of the provided list.
    """
    numbers = list(args)
    numbers.sort()
    length = len(numbers)

    if length % 2 == 0:
        return (numbers[length // 2 - 1] + numbers[length // 2]) / 2
    else:
        return numbers[(length + 1) // 2 - 1]


def ft_quartiles(args: list[int | float]):
    """Returns the first and third quartiles of the provided list.
    """
    numbers = list(args)
    numbers.sort()
    median = ft_median(numbers)

    first = ft_median(filter(lambda x: x <= median, numbers))
    third = ft_median(filter(lambda x: x >= median, numbers))

    return [first, third]


def ft_25(args: list[int | float]):
    """Returns the first quartile of the provided list.
    """
    return ft_quartiles(args)[0]


def ft_75(args: list[int | float]):
    """Returns the third quartile of the provided list.
    """
    return ft_quartiles(args)[1]


def ft_std(args: list[int | float]):
    """Returns the standard deviation of the provided list.
    """
    mean = ft_mean(args)
    partial_sum = [(x - mean) ** 2 for x in args]
    return (sum(partial_sum) / len(args)) ** .5


def ft_var(args: list[int | float]):
    """Returns the variance of the provided list.
    """
    mean = ft_mean(args)
    partial_sum = [(x - mean) ** 2 for x in args]
    return sum(partial_sum) / len(args)


def ft_min(args: list[int | float]):
    """Returns the minimum number in the provided list.
    """
    numbers = list(args)
    numbers.sort()
    return numbers[0]


def ft_max(args: list[int | float]):
    """Returns the maximum number in the provided list.
    """
    numbers = list(args)
    numbers.sort()
    return numbers[len(numbers) - 1]
