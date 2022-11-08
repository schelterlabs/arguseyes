import mlflow
import pickle


class DataLeakageRetrospective:

    def __init__(self, pipeline_run):
        self.pipeline_run = pipeline_run
        self.mlflow_run = mlflow.get_run(run_id=pipeline_run.run.info.run_id)

    def compute_leaked_tuples(self):
        path = f'{self.mlflow_run.info.artifact_uri}/{self.mlflow_run.data.tags["arguseyes.data_leakage.provenance_file"]}'

        with open(path, 'rb') as handle:
            leaked_tuples_provenance = pickle.load(handle)

        inputs_required = set()
        for polynomial in leaked_tuples_provenance:
            for elem in polynomial:
                inputs_required.add(elem.operator_id)

        if len(inputs_required) > 1:
            raise ValueError('Reconstruction of leaked tuples from multiple inputs not implemented yet.')

        leaked_tuple_ids = set()
        for polynomial in leaked_tuples_provenance:
            for elem in polynomial:
                leaked_tuple_ids.add(elem.row_id)

        input_index = list(inputs_required)[0]

        data = self.pipeline_run.load_input_with_provenance(input_index)
        has_been_leaked = lambda p: [polynomial[0]['row_id'] in leaked_tuple_ids for polynomial in p]

        leaked_data = data[has_been_leaked(data['mlinspect_lineage'])]
        leaked_data = leaked_data.drop(columns=['mlinspect_lineage'])
        return leaked_data
