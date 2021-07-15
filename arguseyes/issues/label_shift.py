from scipy import stats


def detect(classification_pipeline) -> bool:
    # TODO this assumes binary classification at the moment, needs to be generalised
    num_pos_train = classification_pipeline.y_train.sum()
    num_neg_train = len(classification_pipeline.y_train) - num_pos_train

    num_pos_test = classification_pipeline.y_test.sum()
    num_neg_test = len(classification_pipeline.y_test) - num_pos_test

    _, bbse_hard_p_val, _, _ = stats.chi2_contingency([[num_pos_train, num_neg_train], [num_pos_test, num_neg_test]])
    label_shift = bbse_hard_p_val < 0.01

    if label_shift:
        print("Label shift between train and test?", label_shift, bbse_hard_p_val)

    return label_shift
