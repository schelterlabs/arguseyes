from arguseyes.refinements import Refinement

from arguseyes.templates import Source, Output

class InputUsage(Refinement):

    @staticmethod
    def _is_used(polynomial, lineage_by_source):
        for entry in polynomial:
            if entry.operator_id in lineage_by_source:
                if entry.row_id in lineage_by_source[entry.operator_id]:
                    return True
        return False

    def _compute(self, pipeline):
        lineage_X_train = pipeline.output_lineage[Output.X_TRAIN]

        lineage_by_source = {}

        for polynomial in lineage_X_train:
            for entry in polynomial:
                if entry.operator_id not in lineage_by_source:
                    lineage_by_source[entry.operator_id] = set()

                lineage_by_source[entry.operator_id].add(entry.row_id)

        refined_sources = []

        for index, source in enumerate(pipeline.train_sources):
            data = source.data

            source_lineage = pipeline.train_source_lineage[index]

            for row_index, row in data.iterrows():                
                data.at[row_index, '__arguseyes__is_used'] = self._is_used(source_lineage[row_index], lineage_by_source)

            refined_source = Source(source.operator_id, source.source_type, data)

            self.log_tag(f'arguseyes.input_usage.source.{index}.operator_id', source.operator_id)
            self.log_tag(f'arguseyes.input_usage.source.{index}.source_type', source.source_type)
            self.log_as_parquet_file(data, f'input-{index}-with-usage.parquet')

            refined_sources.append(refined_source)

        return refined_sources
