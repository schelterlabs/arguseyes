from arguseyes.extraction.with_mlinspect import from_storage
from arguseyes.templates import Output

import mlflow
import pandas as pd
import networkx as nx
import ipycytoscape
from IPython.display import Markdown as md

import pickle


class PipelineRun:

    def __init__(self, run_id):
        self.run = mlflow.get_run(run_id=run_id)
        self.pipeline = from_storage(run_id)

    def _load_dag(self):
        path = f'{self.run.info.artifact_uri}/arguseyes-dag.gpickle'
        dag_file = open(path, "rb")
        dag = pickle.load(dag_file)
        dag_file.close()
        return dag

    def _load_output_with_polynomials(self, output):
        matrix = self.pipeline.outputs[output]
        provenance = self.pipeline.output_lineage[output]

        prov_strs = []
        for polynomial in provenance:
            ids = [f'({lineage_id.operator_id},{lineage_id.row_id})' for lineage_id in polynomial]
            ids.sort()
            prov_strs.append('*'.join(ids))

        return matrix, prov_strs

    def load_X_train(self):
        return self._load_output_with_polynomials(Output.X_TRAIN)

    def load_y_train(self):
        return self._load_output_with_polynomials(Output.Y_TRAIN)

    def load_X_test(self):
        return self._load_output_with_polynomials(Output.X_TEST)

    def load_y_test(self):
        return self._load_output_with_polynomials(Output.Y_TEST)

    def load_input(self, index):
        path = f'{self.run.info.artifact_uri}/arguseyes-dagnode-{index}-lineage-df.parquet'
        data_with_provenance = pd.read_parquet(path)

        provenance = list(data_with_provenance['mlinspect_lineage'])

        prov_strs = [f'({elem[0]["operator_id"]},{elem[0]["row_id"]})' for elem in provenance]

        columns = [column for column in data_with_provenance.columns if column != 'mlinspect_lineage']

        return data_with_provenance[columns], prov_strs

    def load_input_with_provenance(self, index):
        path = f'{self.run.info.artifact_uri}/arguseyes-dagnode-{index}-lineage-df.parquet'
        return pd.read_parquet(path)

    def show_source_code(self):
        source_code = self.run.data.params['arguseyes.pipeline_source']
        return md(f"```Python\n{source_code}\n```")

    def show_plan(self):
        dag = self._load_dag()

        plan = nx.DiGraph()

        data_color = '#355C7D'
        feature_encoding_color = '#C06C84'
        model_color = '#F67280'

        for node in dag.nodes:
            node_id = node.node_id
            operator_type = str(node.operator_info.operator).split('.')[1]

            operator_name = operator_type
            color = '#6C5B7B'

            if operator_type == 'JOIN':
                operator_name = '⋈'
            if operator_type == 'PROJECTION' or operator_type == 'PROJECTION_MODIFY':
                operator_name = 'π'
            if operator_type == 'TRANSFORMER':
                operator_name = 'π'
                color = feature_encoding_color
            if operator_type == 'SELECTION':
                operator_name = 'σ'
            if operator_type == 'CONCATENATION':
                operator_name = '+'
                color = feature_encoding_color
            if operator_type == 'DATA_SOURCE':
                operator_name = f'({node.node_id}) {node.details.description}'
                color = data_color
            if operator_type == 'ESTIMATOR':
                operator_name = 'Model Training'
                color = model_color
            if operator_type == 'SCORE':
                operator_name = 'Model Evaluation'
                color = model_color
            if operator_type in ['TRAIN_DATA', 'TRAIN_LABELS', 'TEST_DATA', 'TEST_LABELS']:
                color = data_color
            if operator_type == 'TRAIN_DATA':
                operator_name = 'X_train'
            if operator_type == 'TRAIN_LABELS':
                operator_name = 'y_train'
            if operator_type == 'TEST_DATA':
                operator_name = 'X_test'
            if operator_type == 'TEST_LABELS':
                operator_name = 'y_test'

            plan.add_node(node_id, operator_name=operator_name, color=color)

        for edge in dag.edges:
            plan.add_edge(edge[0].node_id, edge[1].node_id)

        cytoscapeobj = ipycytoscape.CytoscapeWidget()
        cytoscapeobj.graph.add_graph_from_networkx(plan, directed=True)

        # klay
        cytoscapeobj.set_layout(name='dagre')
        cytoscapeobj.set_style([{
            'selector': 'node',
            'css': {
                'content': 'data(operator_name)',
                'text-valign': 'center',
                'color': 'white',
                'text-outline-width': 2,
                'text-outline-color': 'data(color)',
                'background-color': 'data(color)'
                }
            },
            {
                'selector': ':selected',
                'css': {
                    'background-color': 'black',
                    'line-color': 'black',
                    'target-arrow-color': 'black',
                    'source-arrow-color': 'black',
                    'text-outline-color': 'black'
                }
            },
            {
                "selector": "edge",
                "style": {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle'
                }
            },
        ])

        return cytoscapeobj
