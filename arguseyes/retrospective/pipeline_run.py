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

    def explore_data(self):
        dag = self._load_dag()

        plan = nx.DiGraph()

        #data_color = '#355C7D'
        #feature_encoding_color = '#C06C84'
        #model_color = '#F67280'
        data_color = '#C06C84'
        model_color = '#355C7D'
        feature_encoding_color = '#355C7D'

        for node in dag.nodes:
            node_id = node.node_id
            operator_type = str(node.operator_info.operator).split('.')[1]

            operator_name = operator_type
            #color = '#6C5B7B'
            color = '#355C7D'

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

        while True:
            nodes_to_remove = [node for node, data in plan.nodes(data=True)
                               if data['operator_name'] == 'π' and len(list(plan.successors(node))) == 0]
            if len(nodes_to_remove) == 0:
                break
            else:
                plan.remove_nodes_from(nodes_to_remove)

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

        from IPython.display import Markdown, display
        from ipywidgets import Output, HBox
        import numpy as np

        out = Output(layout={'border': '1px solid black', 'width': '1400px'})

        def log_clicks(display_node):
            node_id = display_node['data']['id']
            dag = self._load_dag()

            dag_node = None
            for node in dag.nodes:
                if int(node.node_id) == int(node_id):
                    dag_node = node

            out.clear_output()
            with out:
                operator_type = str(dag_node.operator_info.operator).split('.')[1]
                if operator_type == 'DATA_SOURCE':
                    md = f'### Input {node_id}\n'

                    line = dag_node.optional_code_info.code_reference.lineno
                    md += f' * Originating from line {line} in the source code:\n'
                    code_lines = ['    ' + line for line in dag_node.optional_code_info.source_code.split('\n')]
                    code_lines = '\n'.join(code_lines)

                    md += "```Python\n"
                    md += code_lines
                    md += "\n```"

                    md += '\n * Load the data and provenance of this input as follows:\n'
                    md += f'```Python\n    data, provenance = run.load_input({node_id})\n```\n'

                    display(Markdown(md))

                    data, _ = self.load_input(node_id)
                    display(data)

                if operator_type == 'TRAIN_DATA':
                    md = f'### Training feature matrix of the pipeline\n'

                    matrix, _ = self.load_X_train()

                    md += f" * Number of rows {matrix.shape[0]}\n"
                    md += f" * Number of columns {matrix.shape[1]}\n"

                    md += '\n * Load this matrix with its provenance as follows:\n'
                    md += f'```Python\n X_train, prov = run.load_X_train()\n```\n'

                    display(Markdown(md))
                    display(matrix)

                if operator_type == 'TRAIN_LABELS':
                    md = f'### Training labels of the pipeline\n'

                    matrix, _ = self.load_y_train()

                    md += f" * Number of entries {matrix.shape[0]}\n"
                    md += f" * Mean value {round(np.mean(matrix), 3)}\n"

                    md += '\n * Load this matrix with its provenance as follows:\n'
                    md += f'```Python\n Xy_train, prov = run.load_y_train()\n```\n'

                    display(Markdown(md))
                    display(matrix)

                if operator_type == 'TEST_DATA':
                    md = f'### Test feature matrix of the pipeline\n'

                    matrix, _ = self.load_X_test()

                    md += f" * Number of rows {matrix.shape[0]}\n"
                    md += f" * Number of columns {matrix.shape[1]}\n"

                    md += '\n * Load this matrix with its provenance as follows:\n'
                    md += f'```Python\n X_test, prov = run.load_X_test()\n```\n'

                    display(Markdown(md))
                    display(matrix)

                if operator_type == 'TEST_LABELS':
                    md = f'### Test labels of the pipeline\n'

                    matrix, _ = self.load_y_test()

                    md += f" * Number of entries {matrix.shape[0]}\n"
                    md += f" * Mean value {round(np.mean(matrix), 3)}\n"

                    md += '\n * Load this matrix with its provenance as follows:\n'
                    md += f'```Python\n Xy_test, prov = run.load_y_test()\n```\n'

                    display(Markdown(md))
                    display(matrix)

        cytoscapeobj.on('node', 'click', log_clicks)

        sidebyside = HBox([cytoscapeobj, out])
        display(Markdown('# Pipeline Data Explorer'))
        display(sidebyside)
        #return cytoscapeobj
