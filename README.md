# ArgusEyes

ArgusEyes is a research prototype for an ML CI server, heavily using [mlinspect](https://github.com/stefan-grafberger/mlinspect).

## Example

__Execute a binary classification pipeline__

Pipelines are written using pandas and sklearn, without manual annotation or instrumentation, checkout this [example for income classification](https://github.com/schelterlabs/argos/blob/main/argos/example_pipelines/income_classifier.py).

```python
from arguseyes.templates.classification import ClassificationPipeline

pipeline = ClassificationPipeline.from_py_file('argos/example_pipelines/income_classifier.py')
```

__Automatically detect violations of best practices in ML__
```python
from arguseyes.issues import train_test_overlap, unnormalised_features

train_test_overlap.detect(pipeline)
unnormalised_features.detect(pipeline)
```

__Automatically detect distribution shift__
```python
from arguseyes.issues import label_shift, covariate_shift

label_shift.detect(pipeline)
covariate_shift.detect(pipeline)
```

__Compute metadata for the pipeline inputs, e.g., whether a given record is used by the model or not__
```python
from arguseyes.refinements import record_lineage

input_df_with_usage = record_lineage.refine(pipeline)[0]

unused_records = input_df_with_usage[input_df_with_usage['__argos__is_used'] == False]
```

## Local setup

Prerequisite: Python 3.9

1. Clone this repository
2. Set up the environment

	`cd argos` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>

3. Install graphviz

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>
	
4. Install pip dependencies 

    `pip install -e .` <br>

5. To ensure everything works, you can run the tests

    `python setup.py test` <br>
