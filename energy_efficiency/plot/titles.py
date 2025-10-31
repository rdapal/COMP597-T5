from enum import Enum

'''
Helpers for titles, axis labels, legends.
'''


def _label_with_unit(obj):
    if hasattr(obj, "get_unit"):
        u = obj.get_unit()
        return f"{obj.get_label()} ({u})"
    return obj.get_label()

def _title_suffix(experiments):
    """
    Build '(MODEL model, TRAINER trainer, N experts, batch size B, G gpus)' using only attributes that
    are invariant across all provided experimets.
    
    e.g: '(switch model, custom trainer, 128 experts, batch size 8, 4 gpus)' -> single experiment
    e.g: '(qwen_moe model, 128 experts, batch size 4, 8 gpus)' -> multiple experiments, varying the trainers
    e.g: '(switch model, deepspeed trainer, 128 experts, 8 gpus)' -> multiple experiments, varying the batch size
    """
    
    # Attributes to include in the title suffix
    attributes = ["model", "trainer", "num_experts", "batch_size", "num_gpus"]

    # Retrieve attributes that are the same across all experiments
    invariant = {}
    for attr in attributes:
        # Skip num_gpus attribute for experiments with simple trainer. They only run on one GPU, but num_gpus can still be invariant across the other experiments
        vals = {getattr(exp, attr) for exp in experiments if not (exp.trainer == Trainer.SIMPLE and attr == "num_gpus")} # set removes duplicates
        if len(vals) == 1:  # if there is only one value, that means this value is the same across all experiments
            invariant[attr] = vals.pop()
    
    # Compose the suffix string
    parts = []
    if "model" in invariant:
        parts.append(f"{invariant['model'].get_label()} model")
    if "trainer" in invariant:
        parts.append(f"{invariant['trainer'].get_label()} trainer")
    if "num_experts" in invariant:
        parts.append(f"{invariant['num_experts']} experts")
    if "batch_size" in invariant:
        parts.append(f"batch size {invariant['batch_size']}")
    if "num_gpus" in invariant:
        parts.append(f"{invariant['num_gpus']} gpus")

    if parts: 
        return f" ({', '.join(parts)})"
    
    return "" # <- this case should not happen

def title_metric_per_iteration(experiments, metric, grouping_enum_type = None):
    base = f"{_label_with_unit(metric)} per iteration"
    if grouping_enum_type is not None:
        base += f" depending on {grouping_enum_type.__name__}"
    return f"{base} {_title_suffix(experiments)}".strip()

def title_dual_metrics_per_iteration(experiments, left_metric, right_metric): 
    base = f"{_label_with_unit(left_metric)} and {_label_with_unit(right_metric)} per iteration"
    return f"{base} {_title_suffix(experiments)}".strip()

def title_metric_vs_metric(experiments, y_metric, x_metric):
    base = f"{_label_with_unit(y_metric)} vs {_label_with_unit(x_metric)}"
    return f"{base} {_title_suffix(experiments)}".strip()

def legend_for_grouping_attribute(experiments, grouping_attribute):
    attr_name = grouping_attribute..__name__.lower()
    members = {getattr(exp, attr_name) for exp in experiments}
    labels = [f"{member.get_abbrev()}: {member.get_label()}" for member in members]
    legend_text = '    '.join(sorted(labels)) #sorted() ensures consistent order
    return legend_text
    
def legend_for_substeps(): 
    return ['Forward', 'Backward', 'Optimiser']


