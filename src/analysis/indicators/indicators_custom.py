#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July 28 20:08:20 2023

@author dhaneor
"""
import bottleneck as bn
import numpy as np
from numba import njit
from indicator import IIndicator, PlotDescription
from indicator_parameter import Parameter

# defining parameters for the Noise indicator
timeperiod = Parameter(
    name="timeperiod", initial_value=14, hard_min=7, hard_max=365, step=7
)



class EfficiencyRatio(IIndicator):
    def __init__(self):
        super().__init__()
        self._name = "ER"
        self._update_name = True
        self.input = ["close"]
        self.output = ["er"]
        self.output_flags = {}

        self._parameters = timeperiod,

        self._is_subplot = True
        self._method: int = 0  # calculation method (0=numpy, 1=numba)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates the (Kaufman) Efficiency Ratio.

        The Efficiency Ratio is a technical indicator used to measure
        the noise in the market. It is dtermined by calculating the
        ratio between:
        a) the absolute price difference between the current price
        and the previous price at the beginning of the lookback period
        b) the sum of the price differences for all lookback periods


        Parameters
        ----------
        data : np.ndarray
            A one- or two-dimensional array of price data.

        Returns
        -------
        np.ndarray
            The Noise indicator values for the lookback set
            in the 'timeperiod' parameter (default: 14).
        """
        if data.ndim == 1:
            if self._method:
                return self._noise_index_numba(data)
            else:
                return self._noise_index_numpy(data)
        elif data.ndim == 2:
            return np.apply_along_axis(
                self._noise_index_numpy
                if self._method == 0
                else self._noise_index_numba, 0, data
                )
        else:
            raise ValueError("Input data must be either 1D or 2D array")

    def help(self):
        return self.run.__doc__

    def plot_description(self) -> PlotDescription:
        return PlotDescription(
            label=self.unique_name,
            is_subplot=self._is_subplot,
            lines=[(self.unique_name, "Line")],
            level="indicator",
        )

    @njit
    def _noise_index_numba(self, data: np.ndarray) -> np.ndarray:
        lookback = self._parameters[0].value

        n = len(data)
        noise = np.empty(n)
        noise[:] = np.nan

        for i in range(lookback, n):
            returns = np.abs(
                np.diff(data[i - lookback: i]) / data[i - lookback: i - 1]
            )
            rolling_sum = np.sum(returns)
            price_diff = np.abs(data[i] / data[i - lookback] - 1)
            noise[i] = price_diff / rolling_sum

        return noise

    def _noise_index_numpy(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the Noise index using NumPy operations.

        This method computes the Noise index, which measures the ratio
        of the net price movement to the sum of absolute price
        movements over a specified lookback period.

        Parameters:
        -----------
        data : np.ndarray
            A 1-dimensional numpy array of price data.

        Returns:
        --------
        np.ndarray
            An array of Noise index values, with the same length as the
            input data. The first 'lookback' elements are padded with
            NaN values.

        Notes:
        ------
        The Noise index is calculated as:
        (abs(final_price / initial_price - 1)) / (sum of absolute returns)
        """
        lookback = self._parameters[0].value

        returns = np.abs(np.diff(data) / data[:-1])
        rolling_sum = bn.move_sum(returns, window=lookback)

        price_diff = np.abs(data[lookback:] / data[:-lookback] - 1)

        noise = price_diff / rolling_sum[lookback - 1 :]

        # Pad the beginning of the array with NaNs to match the input length
        return np.pad(
            noise, (len(data) - len(noise), 0), mode="constant", constant_values=np.nan
        )


custom_indicators = {ind.name: ind for ind in (
    EfficiencyRatio
    )
}


# ====================================================================================
if __name__ == "__main__":
    ind = EfficiencyRatio()
    print(ind.unique_name)
    print(ind.help())

    # define test data with a length of 100
    data = np.random.rand(10_000)

    # define a 2D array with the same data
    data_2d = np.stack((data, data + 0.1), axis=1)

    # run the indicator with a 2D array and print the result
    result_2d = ind.run(data_2d)
    print(result_2d[-50:, 0])
    print(ind.plot_description())

    # run the indicator with a 1D array and print the result
    result_1d = ind.run(data)
    print(result_1d[-50:])
    print(ind.plot_description())
