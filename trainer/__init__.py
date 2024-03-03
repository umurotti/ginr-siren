from .trainer_stage_inr import Trainer as TrainerINR
from .trainer_stage_meta_inr import Trainer as TrainerMetaINR
STAGE_META_INR_ARCH_TYPE = ["meta_low_rank_modulated_inr"]
STAGE_INR_ARCH_TYPE = ["transinr", "low_rank_modulated_transinr"]

def create_trainer(config):
    if config.arch.type in STAGE_META_INR_ARCH_TYPE:
        return TrainerMetaINR
    if config.arch.type in STAGE_INR_ARCH_TYPE:
        return TrainerINR
    else:
        raise ValueError("architecture type not supported")
