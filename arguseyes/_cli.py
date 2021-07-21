import click

from arguseyes import ArgusEyes
from arguseyes.refinements import FairnessMetrics
from arguseyes.issues import  LabelShift, CovariateShift, ConstantFeatures,\
    UnnormalisedFeatures, TrainTestOverlap

@click.command()
@click.argument('series_id')
@click.argument('artifact_storage_uri')
@click.option('-p', '--pipeline', 'pipeline_filename', help='Python file with classification pipeline definition')
def cli(series_id, artifact_storage_uri, pipeline_filename):
    """Run Argus Eyes on your classiciation pipeline defined in Python."""
    click.echo(f'Initializing experiment with ID {series_id}')
    eyes = ArgusEyes(series_id, artifact_storage_uri)

    click.echo(f'Checking fairness metrics for pipeline {pipeline_filename}')
    with eyes.classification_pipeline_from_py_file(pipeline_filename) as pipeline:
        pipeline.compute(FairnessMetrics(sensitive_attribute='sex', non_protected_class='Male'))
        pipeline.compute(FairnessMetrics(sensitive_attribute='race', non_protected_class='White'))
