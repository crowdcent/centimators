import narwhals as nw


# Helper functions for horizontal statistics using narwhals expressions
def var_horizontal(*exprs: nw.Expr, ddof: int = 1) -> nw.Expr:
    """Computes the variance horizontally (row-wise) across a set of expressions.

    Args:
        *exprs (nw.Expr): Narwhals expressions representing the columns to compute variance over.
        ddof (int, default=1): Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.

    Returns:
        nw.Expr: A Narwhals expression for the horizontal variance.
    """
    actual_exprs = list(exprs)
    n = len(actual_exprs)

    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    sum_sq_diff_expr = nw.sum_horizontal(
        *[(expr - mean_expr) ** 2 for expr in actual_exprs]
    )

    denominator = n - ddof
    if denominator <= 0:
        # Variance is undefined or NaN (e.g., single point with ddof=1)
        return nw.lit(float("nan"), dtype=nw.Float64)

    return sum_sq_diff_expr / nw.lit(denominator, dtype=nw.Float64)


def std_horizontal(*exprs: nw.Expr, ddof: int = 1) -> nw.Expr:
    """Computes the standard deviation horizontally (row-wise) across a set of expressions.

    Args:
        *exprs (nw.Expr): Narwhals expressions representing the columns to compute standard deviation over.
        ddof (int, default=1): Delta Degrees of Freedom. The divisor used in calculations is N - ddof.

    Returns:
        nw.Expr: A Narwhals expression for the horizontal standard deviation.
    """
    actual_exprs = list(exprs)
    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    variance_expr = var_horizontal(*actual_exprs, ddof=ddof)
    # sqrt of NaN is NaN; sqrt of negative (float precision issues) also leads to NaN in backends
    return variance_expr**0.5


def skew_horizontal(*exprs: nw.Expr) -> nw.Expr:
    """Computes the skewness horizontally (row-wise) across a set of expressions.

    Uses a bias-corrected formula.

    Args:
        *exprs (nw.Expr): Narwhals expressions representing the columns to compute skewness over.

    Returns:
        nw.Expr: A Narwhals expression for the horizontal skewness.
    """
    actual_exprs = list(exprs)
    n = len(actual_exprs)

    if n < 3:
        # Skewness with this specific correction factor is undefined for n < 3
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    # ddof=1 for sample standard deviation is standard in skewness calculations
    std_dev_expr = std_horizontal(*actual_exprs, ddof=1)

    # Calculate sum of ((expr - mean) / std_dev)**3
    # This relies on (0/0) -> NaN propagation if std_dev_expr is 0.
    # If std_dev_expr is 0, all (expr - mean_expr) must also be 0 for finite mean.
    # Then (0/0)**3 is NaN. Sum of NaNs is NaN.
    standardized_cubed_deviations = [
        ((expr - mean_expr) / std_dev_expr) ** 3 for expr in actual_exprs
    ]
    sum_std_cubed = nw.sum_horizontal(*standardized_cubed_deviations)

    # Bias correction factor: n / ((n - 1) * (n - 2))
    correction_factor_val = n / ((n - 1) * (n - 2))

    # If std_dev_expr was 0, sum_std_cubed is NaN. NaN * factor is NaN.
    return sum_std_cubed * nw.lit(correction_factor_val, dtype=nw.Float64)


def kurtosis_horizontal(*exprs: nw.Expr) -> nw.Expr:
    """Computes the excess kurtosis (Fisher's g2) horizontally (row-wise)
    across a set of expressions. Uses a bias-corrected formula.

    Excess kurtosis indicates how much the tails of the distribution differ
    from the tails of a normal distribution. Positive values indicate heavier
    tails (leptokurtic), negative values indicate lighter tails (platykurtic).

    The formula for the sample excess kurtosis (G2) is used:
    G2 = { [n(n+1)] / [(n-1)(n-2)(n-3)] } * sum[ ( (x_i - mean) / std_sample )^4 ]
         - { [3(n-1)^2] / [(n-2)(n-3)] }
    This is undefined for n < 4.

    Args:
        *exprs (nw.Expr): Narwhals expressions representing the columns to compute kurtosis over.

    Returns:
        nw.Expr: A Narwhals expression for the horizontal excess kurtosis.
    """
    actual_exprs = list(exprs)
    n = len(actual_exprs)

    if n < 4:
        # Kurtosis with this specific correction factor is undefined for n < 4
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    # ddof=1 for sample standard deviation is standard in this kurtosis formula
    std_dev_expr = std_horizontal(*actual_exprs, ddof=1)

    # Calculate sum of ((expr - mean) / std_dev)**4
    # If std_dev_expr is 0 (constant data), (0/0) -> NaN. Sum of NaNs is NaN. Correct.
    standardized_fourth_powers = [
        ((expr - mean_expr) / std_dev_expr) ** 4 for expr in actual_exprs
    ]
    sum_std_fourth = nw.sum_horizontal(*standardized_fourth_powers)

    # Bias correction terms
    term1_coeff_val = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    term2_val = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    # If sum_std_fourth is NaN, the result will be NaN.
    return (sum_std_fourth * nw.lit(term1_coeff_val, dtype=nw.Float64)) - nw.lit(
        term2_val, dtype=nw.Float64
    )


def range_horizontal(*exprs: nw.Expr) -> nw.Expr:
    """Computes the range (max - min) horizontally (row-wise) across a set of expressions.

    Args:
        *exprs (nw.Expr): Narwhals expressions representing the columns to compute range over.

    Returns:
        nw.Expr: A Narwhals expression for the horizontal range.
    """
    actual_exprs = list(exprs)

    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    min_val = nw.min_horizontal(*actual_exprs)
    max_val = nw.max_horizontal(*actual_exprs)

    return max_val - min_val


def coefficient_of_variation_horizontal(*exprs: nw.Expr, ddof: int = 1) -> nw.Expr:
    """Computes the coefficient of variation (CV) horizontally (row-wise)
    across a set of expressions.

    CV = standard_deviation / mean

    Args:
        *exprs (nw.Expr): Narwhals expressions representing the columns to compute CV over.
        ddof (int, default=1): Delta Degrees of Freedom for the standard deviation calculation.

    Returns:
        nw.Expr: A Narwhals expression for the horizontal coefficient of variation.
            Returns NaN if mean is zero and std is zero.
            Returns Inf or -Inf if mean is zero and std is non-zero.
    """
    actual_exprs = list(exprs)

    if not actual_exprs:
        return nw.lit(float("nan"), dtype=nw.Float64)

    mean_expr = nw.mean_horizontal(*actual_exprs)
    std_expr = std_horizontal(*actual_exprs, ddof=ddof)

    # Division handles cases:
    # std/0 where std is non-zero -> inf
    # 0/0 -> NaN
    # std/mean
    return std_expr / mean_expr


# Example usage:
# var_expr = var_horizontal(nw.Float64(1), nw.Float64(2), nw.Float64(3))
# std_expr = std_horizontal(nw.Float64(1), nw.Float64(2), nw.Float64(3))
# skew_expr = skew_horizontal(nw.Float64(1), nw.Float64(2), nw.Float64(3))
# kurtosis_expr = kurtosis_horizontal(nw.Float64(1), nw.Float64(2), nw.Float64(3))
# range_expr = range_horizontal(nw.Float64(1), nw.Float64(2), nw.Float64(3))
# cv_expr = coefficient_of_variation_horizontal(nw.Float64(1), nw.Float64(2), nw.Float64(3))
