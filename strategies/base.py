from abc import ABC, abstractmethod


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.

    A sampling strategy defines a rule to generate a random sample
    from a probability distribution or stochastic process.

    In mathematical terms, a concrete implementation of this class
    represents a mapping:

        sample() : Ω → ℝ

    where Ω is the underlying probability space and the returned value
    is a realization of a random variable.

    This class follows the Strategy design pattern, allowing different
    sampling methods (e.g. inverse CDF, rejection sampling, importance
    sampling) to be used interchangeably.
    """

    @abstractmethod
    def sample(self) -> float:
        """
        Generate a single sample.

        Returns:
            float: A realization of the random variable defined by
                   the specific sampling strategy.
        """
        # Must be implemented by concrete subclasses
        pass
