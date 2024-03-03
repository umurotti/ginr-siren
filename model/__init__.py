
from .generalizable_INR import transinr,  low_rank_modulated_transinr, meta_low_rank_modulated_inr


def create_model(config):
    model_type = config.type.lower()
    if model_type == "transinr":
        model = transinr(config)
    elif model_type == "low_rank_modulated_transinr":
        model = low_rank_modulated_transinr(config)

    elif model_type == "meta_low_rank_modulated_inr":
        model = meta_low_rank_modulated_inr(config)

    else:
        raise ValueError(f"{model_type} is invalid..")

    return model
