
## ArgusEyes

The idea of ArgusEyes is to use [mlinspect](https://github.com/stefan-grafberger/mlinspect) to extract the intermediate results and their lineage from native ML pipelines, and enable a set of automated issue detection and data refinement techniques.

<img src="https://github.com/schelterlabs/arguseyes/blob/master/core-idea.png">

## We provide a couple of example pipelines and notebooks to showcase the usage:

* [Checking for common issues in ML](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/demo-issues-in-example.ipynb) such as label shift, covariate shift, unnormalised features etc. in our [example pipeline](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/paper-example.py) 
 * [Determining which tuples from the raw input data are used](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/demo-reviews-usage.ipynb) to train the classifier in our [Amazon Reviews pipeline](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/amazon-reviews.py).
 * [Computing group fairness metrics](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/demo-income-fairness.ipynb) for an [income-level classification pipeline](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/income-classifier.py).
 * [Estimating the value of input samples](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/demo-sneakers-valuation.ipynb) (via Shapley values) for a [shoe image classification pipeline](https://github.com/schelterlabs/arguseyes/blob/master/arguseyes/example_pipelines/sneakers.py).

Note that you can run `mlflow ui --backend-store-uri ./mlruns` from the project root to already view some of the captured artifacts and issue detection results.

## Local setup

Prerequisite: Python 3.9

1. Clone this repository
2. Set up the environment

	`cd arguseyes` <br>
	`python -m venv venv` <br>
	`source venv/bin/activate` <br>

3. Install graphviz

    `Linux: ` `apt-get install graphviz` <br>
    `MAC OS: ` `brew install graphviz` <br>
	
4. Install pip dependencies 

    `pip install -r requirements.txt` <br>

