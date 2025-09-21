# V2 Failure Analysis: 0% Precision Cases

## Summary
Analysis of 47 cases where V2 achieved 0% precision in bug localization.

## Key Findings

### 1. Missing the Ground Truth File Entirely (68% of failures)
- **32 out of 47 cases** never even viewed the correct file
- The model's search strategy fails to discover the actual bug location
- Problems identified:
  - Model follows surface-level clues (error messages, symptoms) rather than understanding deeper code flow
  - Initial searches are often too narrow or focused on the wrong area
  - Model doesn't explore alternative hypotheses when the first approach doesn't yield results
- **Example**: `django__django-11964` - Bug is in `django/db/models/enums.py` but model only explored `fields/__init__.py` and `fields/base.py`

### 2. Viewing Ground Truth but Choosing Symptom Over Root Cause (31% of failures)
- **15 out of 47 cases** viewed the correct file but dismissed it
- Key patterns:
  - **Repetitive viewing**: Cases like `django__django-14238` viewed the correct file 8 times but still chose wrong
  - **Symptom vs. cause confusion**: Model identifies where error *manifests* rather than where it needs to be *fixed*
  - **Anchoring bias**: Once model forms initial hypothesis, struggles to reconsider even with contradictory evidence
- **Example**: `django__django-11422` - Model viewed `autoreload.py` (correct) 3 times but kept returning to `runserver.py` (symptom location) 4 times

## Navigation Patterns
- **Narrow search** (single directory): 18 cases
- **Broad search** (multiple directories): 29 cases
- **Repetitive viewing** (same file 3+ times): 18 cases

## Root Cause
The model lacks a **systematic root cause analysis approach**. It tends to:
1. Follow the error trail superficially
2. Confuse where problems *appear* with where they must be *fixed*
3. Get anchored on initial hypotheses without reconsidering

## Recommendations for Improvement
1. **Distinguish between "where the error shows up" vs "where the fix belongs"**
   - Add explicit guidance in system prompt about symptom vs root cause
   - Encourage tracing back from error manifestation to source

2. **Explore more broadly before narrowing down**
   - Encourage initial broad exploration of codebase structure
   - Delay commitment to specific files until sufficient context gathered

3. **Explicitly consider alternative hypotheses when viewing files multiple times**
   - If viewing same file repeatedly, prompt reconsideration of assumptions
   - Encourage exploring other possibilities before returning to same location

## Specific Cases for Reference

### Viewed GT but Chose Wrong:
- `django__django-11422`: autoreload.py → runserver.py
- `django__django-13033`: compiler.py → query.py  
- `django__django-13448`: (multiple files involved)
- `django__django-14238`: fields/__init__.py → options.py
- `django__django-15252`: (multiple files involved)

### Never Viewed GT (Examples):
- `django__django-11964`: Never found enums.py
- `django__django-11564`: Never found conf/__init__.py
- `django__django-11620`: Never found views/debug.py
