import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator


def plot_null_distribution(saved_stats: pd.DataFrame, ind: int, ax=None) -> plt.Axes:
    """
    Plot the reconstructed null distribution of MB-VIP scores for a single variable.

    Parameters
    ----------
    saved_stats : pd.DataFrame
        DataFrame of summary statistics for null distributions, as returned by
        MamsiPls.mb_vip_permtest(). Expected columns: 'observed_vip', 'null_min', 'null_max',
        'null_mean', 'null_std', 'null_skewness', 'null_kurtosis', 'null_perc_25', 'null_median', 'null_perc_75',
        'null_perc_95', 'null_perc_99'.
    ind : int
        Index of the variable to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    obs_vip = saved_stats['observed_vip'][ind]
    x_axis = np.linspace(saved_stats['null_min'][ind], saved_stats['null_max'][ind], 500)

    # Pearson3 (moment-matching)
    pdf_pearson = stats.pearson3.pdf(
        x_axis,
        skew=saved_stats['null_skewness'][ind],
        loc=saved_stats['null_mean'][ind],
        scale=saved_stats['null_std'][ind],
    )

    # Inverse CDF + KDE (percentile-matching)
    x_points = [
        saved_stats['null_min'][ind], saved_stats['null_perc_25'][ind], saved_stats['null_median'][ind],
        saved_stats['null_perc_75'][ind], saved_stats['null_perc_95'][ind],
        saved_stats['null_perc_99'][ind], saved_stats['null_max'][ind],
    ]
    y_points = [0.0, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    inv_cdf_spline = PchipInterpolator(y_points, x_points)
    simulated_vips = inv_cdf_spline(np.linspace(0, 1, 10_000))
    pdf_smooth_percentile = stats.gaussian_kde(simulated_vips)(x_axis)

    # Gamma (strictly positive, right-skewed)
    mean_val = saved_stats['null_mean'][ind]
    var_val = saved_stats['null_std'][ind] ** 2
    pdf_gamma = stats.gamma.pdf(
        x_axis,
        a=(mean_val ** 2) / var_val,
        scale=var_val / mean_val,
    )

    ax.plot(x_axis, pdf_pearson, label='Pearson3 (moment-based)', color='blue', linestyle=':')
    ax.plot(x_axis, pdf_smooth_percentile, label='KDE (percentile-based)', color='orange', linestyle='--')
    ax.plot(x_axis, pdf_gamma, label='Gamma distribution', color='green', linestyle='-.')
    ax.axvline(obs_vip, color='red', label=f'Observed VIP ({obs_vip:.4f})')

    ax.set_title(f'Null Distribution Estimates (Feature: {saved_stats["feature"][ind]}, index: {ind}), p-value: {saved_stats["p_value"][ind]:.4f}')
    ax.set_xlabel('MB-VIP Score')
    ax.set_ylabel('Density')
    ax.legend()

    print(f"Feature: {saved_stats['feature'][ind]}"
          f"\nObserved VIP: {obs_vip:.4f}"
          f"\nNull Mean: {saved_stats['null_mean'][ind]:.4f}"
          f"\nNull Std Dev: {saved_stats['null_std'][ind]:.4f}"
          f"\nNull Skewness: {saved_stats['null_skewness'][ind]:.4f}"
          f"\nNull Kurtosis: {saved_stats['null_kurtosis'][ind]:.4f}"
          f"\nNull 25th Percentile: {saved_stats['null_perc_25'][ind]:.4f}"
          f"\nNull Median: {saved_stats['null_median'][ind]:.4f}"
          f"\nNull 75th Percentile: {saved_stats['null_perc_75'][ind]:.4f}"
          f"\nNull 95th Percentile: {saved_stats['null_perc_95'][ind]:.4f}"
          f"\nNull 99th Percentile: {saved_stats['null_perc_99'][ind]:.4f}"
          f"\nNull Min: {saved_stats['null_min'][ind]:.4f}"
          f"\nNull Max: {saved_stats['null_max'][ind]:.4f}"
          f"\nP-value: {saved_stats['p_value'][ind]:.4f}"
          f"\nExceedance Count: {saved_stats['greater_than_observed'][ind]} out of {saved_stats['n_permutations'][ind]} permutations")
  
           


    return ax
