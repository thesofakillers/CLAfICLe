from typing import Any, Callable, Dict, Tuple

from .xglue import process as xglue

"""
Maps name to processing function.
Each processing function does the following:

Gets relevant test split
Generates k-shot context
Prepends each input with k-shot context
Adds options column to track options
Returns processed test dataset and relevant metrics
"""
process_by_name: Dict[str, Callable[[Any, str, str], Tuple]] = {"xglue;qam": xglue}
