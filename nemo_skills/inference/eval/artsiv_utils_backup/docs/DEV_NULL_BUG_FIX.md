# /dev/null Bug Fix in Ground Truth Extraction

## Issue Description

The `extract_locations_from_patch` function was incorrectly including `/dev/null` as a ground truth file when processing patches that create new files.

## Root Cause

In git diff format, when a new file is created, it shows:
```diff
--- /dev/null
+++ b/astropy/coordinates/builtin_frames/itrs_observed_transforms.py
```

The function was naively using the path from the `---` line, which resulted in `/dev/null` being treated as a file that needs to be modified.

## Impact

This caused:
1. Incorrect ground truth file lists (including `/dev/null`)
2. Potentially skewed metrics when checking if ground truth files exist in the repository
3. Confusion about which files actually need to be modified

## Fix Applied

### 1. Updated `extract_locations_from_patch`
- Now detects when `--- /dev/null` indicates a new file
- Waits for the `+++` line to get the actual file path
- Properly handles new files by marking them at line 1

### 2. Added `extract_files_from_patch`
- New utility function that simply extracts unique file paths from a patch
- Automatically excludes `/dev/null`
- Cleaner API for when you just need the list of files, not line ranges

## Example

For the patch you showed:
```diff
--- a/astropy/coordinates/builtin_frames/__init__.py
+++ b/astropy/coordinates/builtin_frames/__init__.py
...
--- /dev/null
+++ b/astropy/coordinates/builtin_frames/itrs_observed_transforms.py
```

### Before Fix:
```python
ground_truth_files = [
    "astropy/coordinates/builtin_frames/__init__.py",
    "/dev/null",  # WRONG!
    "astropy/coordinates/builtin_frames/itrs_observed_transforms.py",
    ...
]
```

### After Fix:
```python
ground_truth_files = [
    "astropy/coordinates/builtin_frames/__init__.py", 
    "astropy/coordinates/builtin_frames/itrs_observed_transforms.py",  # Correct!
    ...
]
```

## Usage Recommendation

For extracting ground truth files, consider using the new `extract_files_from_patch` function:
```python
from nemo_skills.inference.eval.artsiv_utils.utils import extract_files_from_patch

ground_truth_files = extract_files_from_patch(patch)
```

This is cleaner and more direct than extracting locations and then getting unique file paths.
