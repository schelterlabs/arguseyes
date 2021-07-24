from arguseyes.issues._covariate_shift import evaluate_domain_classifier
from arguseyes.templates.classification import ClassificationPipeline


def detect(artifact_storage_uri, run_ids):
    X_trains = {}
    for run_id in run_ids:
        print(run_id)
        with ClassificationPipeline.from_storage(run_id, artifact_storage_uri) as pipeline:
            X_trains[run_id] = pipeline.X_train

    for index, run_id_a in enumerate(run_ids):
        for run_id_b in run_ids[index + 1:]:
            auc = evaluate_domain_classifier(X_trains[run_id_a], X_trains[run_id_b])
            auc_threshold = 0.7
            covariate_shift = auc > auc_threshold
            if covariate_shift:
                print(f'Found covariate shift between the training features of runs {run_id_a} and {run_id_b}')
