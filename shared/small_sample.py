"""Bounds a sample is too small to support.

Handed a valid one-row CSV, several tools here answered with confidence about
things the row count had already decided:

  detect_outliers      lower_bound == upper_bound == the value, "0 outliers"
  anomaly_detection    IsolationForest fitted on the single row it then scores
  check_data_quality   30/100, every column flagged constant
  batch_predict        min/max/mean over one point, as a prediction distribution

None of those numbers is a measurement. Two of the thresholds below are pure
arithmetic rather than judgement:

  * the 1.5*IQR fence cannot fall inside a sample of fewer than four values --
    for [0, a, M] the upper fence is (a+M)/2 + 0.75*M, which exceeds M for
    every a < M -- so below n=4 nothing can ever be flagged;
  * the largest z-score attainable by any of n points, using the sample
    standard deviation, is (n-1)/sqrt(n). That first exceeds 3 at n=11, so a
    3-sigma scan over ten rows or fewer is guaranteed to report zero outliers
    whatever the data says.

A tool that reports "0 outliers" from either case is describing its input's
size. Say undetermined instead: a missing verdict reads as missing, while a
defaulted one reads as a finding.
"""

from __future__ import annotations

import math
from typing import Any

# See the module docstring for why four, and why eleven.
MIN_N_IQR = 4


def min_n_for_zscore(threshold: float = 3.0) -> int:
    """Smallest n where some point *could* exceed `threshold` standard deviations."""
    if threshold <= 0:
        return 2
    # (n-1)/sqrt(n) > t  <=>  u**2 - t*u - 1 > 0 for u = sqrt(n).
    u = (threshold + math.sqrt(threshold * threshold + 4.0)) / 2.0
    return int(math.floor(u * u)) + 1


def finite(value: Any) -> float | None:
    """A float for reporting, or None if it is NaN or infinite.

    `round(float("nan"), 4)` is still NaN, and json.dumps writes that as the
    bare token `NaN` -- not valid JSON, and read as a number by some clients.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def rounded(value: Any, digits: int = 4) -> float | None:
    """`finite`, then rounded. None stays None."""
    number = finite(value)
    return None if number is None else round(number, digits)
