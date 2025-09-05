import numpy as np
import matplotlib.pyplot as plt
import pytest
from seismostats.plots.statistical import plot_b_series_constant_nm
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def test_plot_b_series_constant_nm():
    delta_m = 0.1
    mc = 1
    magnitudes = simulate_magnitudes_binned(1000, b=1, delta_m=delta_m, mc=mc)
    times = np.arange(1000)

    # --- successful run ---
    ax = plot_b_series_constant_nm(
        magnitudes=magnitudes,
        delta_m=delta_m,
        mc=mc,
        times=times,
        n_m=200,
    )
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)  # cleanup

    # --- error: n_m smaller than min_num ---
    with pytest.raises(ValueError):
        plot_b_series_constant_nm(
            magnitudes=magnitudes,
            delta_m=0.1,
            mc=mc,
            times=times,
            n_m=5,
            min_num=20,
        )

    # --- error: mismatched lengths ---
    with pytest.raises(IndexError):
        plot_b_series_constant_nm(
            magnitudes=magnitudes,
            delta_m=0.1,
            mc=mc,
            times=np.arange(49),  # mismatch
            n_m=20,
        )
