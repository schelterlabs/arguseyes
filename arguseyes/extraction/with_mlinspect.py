import os
import mlflow
from mlflow.tracking import MlflowClient
from PIL import Image
import tempfile
from contextlib import redirect_stdout
from networkx.readwrite.gpickle import read_gpickle, write_gpickle
import pandas as pd
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from mlinspect import PipelineInspector
from mlinspect.inspections import ArgumentCapturing
from mlinspect.inspections._lineage import RowLineage, LineageId
from mlinspect.visualisation import save_fig_to_path
from mlinspect.inspections._inspection_input import OperatorType

from arguseyes.templates.classification import ClassificationPipeline
from arguseyes.extraction import feature_matrix_extractor, source_extractor
from arguseyes.templates import Output, SourceType, Source

def from_py_file(path_to_py_file, cmd_args=[]):
    synthetic_cmd_args = ['eyes']
    synthetic_cmd_args.extend(cmd_args)
    from unittest.mock import patch
    import sys
    logging.info(f'Patching sys.argv with {synthetic_cmd_args}')
    with patch.object(sys, 'argv', synthetic_cmd_args):
        return _execute_pipeline(PipelineInspector.on_pipeline_from_py_file(path_to_py_file))

def from_notebook(path_to_ipynb_file):
    return _execute_pipeline(
        PipelineInspector.on_pipeline_from_ipynb_file(path_to_ipynb_file))



def _execute_pipeline(inspector: PipelineInspector):
    lineage_inspection = RowLineage(RowLineage.ALL_ROWS, [OperatorType.DATA_SOURCE, OperatorType.TRAIN_DATA,
                                                          OperatorType.TRAIN_LABELS, OperatorType.TEST_DATA,
                                                          OperatorType.TEST_LABELS, OperatorType.SCORE,
                                                          OperatorType.JOIN])
    argument_capturing_inspection = ArgumentCapturing()

    mlflow.start_run()
    logging.info(f'Created run {mlflow.active_run().info.run_id} for this invocation')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logging.info('Executing instrumented user pipeline with mlinspect')
    with tempfile.TemporaryDirectory() as tmpdirname:
        logging.info('Redirecting the pipeline\'s stdout to arguseyes-pipeline-output.txt')
        with open(os.path.join(tmpdirname, 'arguseyes-pipeline-output.txt'), 'w') as tmpfile:
            with redirect_stdout(tmpfile):
                result = inspector \
                    .add_required_inspection(lineage_inspection) \
                    .add_required_inspection(argument_capturing_inspection) \
                    .execute()
                mlflow.log_artifact(tmpfile.name)

    captured_arguments = {
        node: node_results[argument_capturing_inspection]
        for node, node_results in result.dag_node_to_inspection_results.items()
    }

    for node, args in captured_arguments.items():
        if len(args) > 0:
            node_id = node.node_id
            mlflow.log_param(f'arguseyes.preprocessing.{node_id}.type',
                             node.operator_info.function_info.function_name)
            mlflow.log_param(f'arguseyes.preprocessing.{node_id}.arguments',
                             args)

    dag_node_to_lineage_df = {
        # node: df
        # for node, lineage_map in result.dag_node_to_inspection_results.items()
        # for df in lineage_map.values()
        node: node_results[lineage_inspection]
        for node, node_results in result.dag_node_to_inspection_results.items()
    }    

    return _from_dag_and_lineage(result.dag, dag_node_to_lineage_df)



def _from_dag_and_lineage(dag, dag_node_to_lineage_df, log_results=True):

    logging.info(f'Identifying training sources')
    train_sources, train_source_lineage = source_extractor.extract_train_sources(dag, dag_node_to_lineage_df)
    logging.info(f'Identifying test sources')
    test_sources, test_source_lineage = source_extractor.extract_test_sources(dag, dag_node_to_lineage_df)

    X_train, lineage_X_train = feature_matrix_extractor.extract_train_feature_matrix(dag_node_to_lineage_df)
    logging.info(f'Extracted feature matrix X_train with {X_train.shape[0]} rows and {X_train.shape[1]} columns')
    X_test, lineage_X_test = feature_matrix_extractor.extract_test_feature_matrix(dag_node_to_lineage_df)
    logging.info(f'Extracted feature matrix X_test with {X_test.shape[0]} rows and {X_test.shape[1]} columns')

    y_train, lineage_y_train = feature_matrix_extractor.extract_train_labels(dag_node_to_lineage_df)
    y_test, lineage_y_test = feature_matrix_extractor.extract_test_labels(dag_node_to_lineage_df)
    y_pred, lineage_y_pred = feature_matrix_extractor.extract_predicted_labels(dag_node_to_lineage_df)
    logging.info(f'Extracted y_train, y_test and y_pred')

    outputs = {
        Output.X_TRAIN: X_train,
        Output.Y_TRAIN: y_train,
        Output.X_TEST: X_test,
        Output.Y_TEST: y_test,
        Output.Y_PRED: y_pred
    }    

    output_lineage = {
        Output.X_TRAIN: lineage_X_train,
        Output.Y_TRAIN: lineage_y_train,
        Output.X_TEST: lineage_X_test,
        Output.Y_TEST: lineage_y_test,
        Output.Y_PRED: lineage_y_pred
    }    

    if log_results:
        _log_mlinspect_results(dag, dag_node_to_lineage_df)

    return ClassificationPipeline(train_sources, train_source_lineage, test_sources, test_source_lineage,
                                  outputs, output_lineage)


def from_storage(run_id, artifact_storage_uri):
    run = MlflowClient(artifact_storage_uri).get_run(run_id)

    # Retrieve pickled DAG (as networkx.DiGraph) with read_gpickle
    dag_filename = os.path.join(run.info.artifact_uri, "arguseyes-dag.gpickle")
    dag = read_gpickle(dag_filename)

    # Map DagNode objects from unpickled DAG object above
    # to DateFrames of lineage inspection results from Parquet files
    # each file named with DagNode.node_id
    dag_node_to_lineage_df = {}
    for node in dag.nodes:
        df_filename = os.path.join(
            run.info.artifact_uri, f"arguseyes-dagnode-{node.node_id}-lineage-df.parquet")
        if not os.path.exists(df_filename):
            continue
        df = pd.read_parquet(df_filename)
        df['mlinspect_lineage'] = df['mlinspect_lineage'].map(
            lambda l: set(LineageId(**item) for item in l))
        dag_node_to_lineage_df[node] = df

    return _from_dag_and_lineage(dag, dag_node_to_lineage_df, log_results=False)



def _log_mlinspect_results(dag, dag_node_to_lineage_df):

    with tempfile.TemporaryDirectory() as tmpdirname:
        dag_filename = os.path.join(tmpdirname, 'arguseyes-dag.png')
        save_fig_to_path(dag, dag_filename)
        dag_image = Image.open(dag_filename).convert("RGB")
        mlflow.log_image(dag_image, 'arguseyes-dag.png')

    with tempfile.TemporaryDirectory() as tmpdirname:
        dag_filename = os.path.join(tmpdirname, 'arguseyes-dag.gpickle')
        write_gpickle(dag, dag_filename)
        mlflow.log_artifact(dag_filename)

    for node, orig_df in dag_node_to_lineage_df.items():
        if orig_df is None:
            continue
        filename = f'arguseyes-dagnode-{node.node_id}-lineage-df.parquet'
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, filename)
            # Currently, pyarrow cannot serialise sets
            lineage_column = orig_df['mlinspect_lineage'].map(
                lambda s: [
                    {
                        'operator_id': lid.operator_id,
                        'row_id': lid.row_id,
                    } for lid in s
                ]
            )
            mod_df = orig_df.drop(columns=['mlinspect_lineage'])
            mod_df['mlinspect_lineage'] = lineage_column
            table = pa.Table.from_pandas(mod_df, preserve_index=True)
            pq.write_table(table, temp_filename)
            mlflow.log_artifact(temp_filename)
