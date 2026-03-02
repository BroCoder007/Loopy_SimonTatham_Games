"""
Merge Sort
==========
Pure deterministic merge sort for the DP solver engine.

Provides a stable, comparison-based sort with O(n log n) worst-case
complexity.  Used in place of built-in sorted() throughout the DP
solver to satisfy the "implement merge sort and use it" requirement.

The implementation handles any sequence whose elements support `<`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def merge_sort(
    seq: Sequence[T],
    *,
    key: Optional[Callable[[T], Any]] = None,
) -> List[T]:
    """
    Return a new list containing items from *seq* in ascending order.

    Parameters
    ----------
    seq : sequence
        Input items (list, tuple, set, dict_keys, …).
    key : callable, optional
        One-argument function used to extract a comparison key from each
        element, identical semantics to ``sorted(…, key=…)``.

    Returns
    -------
    list
        A fresh sorted list.  The original *seq* is never mutated.
    """
    items: List[T] = list(seq)
    n = len(items)
    if n <= 1:
        return items

    mid = n // 2
    left = merge_sort(items[:mid], key=key)
    right = merge_sort(items[mid:], key=key)
    return _merge(left, right, key)


def _merge(
    left: List[T],
    right: List[T],
    key: Optional[Callable[[T], Any]],
) -> List[T]:
    """Merge two sorted lists into one sorted list (stable)."""
    result: List[T] = []
    i = j = 0
    len_l = len(left)
    len_r = len(right)

    while i < len_l and j < len_r:
        lk = key(left[i]) if key is not None else left[i]
        rk = key(right[j]) if key is not None else right[j]
        if lk <= rk:          # stable: equal elements keep left-first order
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append remaining tail
    if i < len_l:
        result.extend(left[i:])
    if j < len_r:
        result.extend(right[j:])

    return result
