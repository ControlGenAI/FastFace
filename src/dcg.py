import torch
import matplotlib.pyplot as plt

from .cfg_schedulers import (
    default_sch_kwargs,
    get_scheduler
)


def decoupled_cfg_predict(
    noise_pred, 
    a, 
    b, 
    step,
    dcg_type=1,
    term_preproc="",
    term_postproc="",
    rescale=0.7,
    a_scheduler="default",
    b_scheduler="default",
    sch_kwargs=default_sch_kwargs,
    return_norms=False,
    *args,
    **kwargs,
):
    out = dict()
    uncond, out1, out2 = noise_pred.chunk(3)
    # NOTE: method and paper correspond to dcg_type == 3
    if dcg_type in [1, 3]:
        # 1) out1, out2 = eps_t, eps_ti
        # 3) out1, out2 = eps_i, eps_ti
        term1 = out1 - uncond
        term2 = out2 - out1
    elif dcg_type == 2:
        # out1, out2 = eps_t, eps_i
        term1 = out1 - uncond
        term2 = out2 - uncond

    if return_norms:
        out["norms"]=[torch.norm(t).cpu().item() for t in (uncond, out1, out2, term1, term2)]

    if "renorm" in term_preproc:
        term2 *= torch.norm(term1) / torch.norm(term2)

    # apply scheduler and pass additional args 
    step_idx, t = step
    if a_scheduler != "custom":
        a = get_scheduler(a_scheduler)(a, t, log=True, **sch_kwargs["a"])
    else:
        custom_sch = get_scheduler("custom")(
            w_values=sch_kwargs["custom"]["a"]
        )
        a = custom_sch(a, step_idx)
    if b_scheduler != "custom":
        b = get_scheduler(b_scheduler)(b, t, log=True, **sch_kwargs["b"])
    else:
        custom_sch = get_scheduler("custom")(
            w_values=sch_kwargs["custom"]["b"]
        )
        b = custom_sch(b, step_idx)
    
    # dcg prediction
    pred = uncond + a * term1 + b * term2

    if term_postproc.startswith("rescale"): # inspired by https://arxiv.org/pdf/2305.08891, Alg. 2
        std1 = out1.std([1, 2, 3], keepdim=True)
        std2 = out2.std([1, 2, 3], keepdim=True)
        std_dcg = pred.std([1, 2, 3], keepdim=True)
        factor = (std1 + std2) / (2 * std_dcg)
        factor = rescale * factor + (1 - rescale)
        pred *= factor

    out["pred"] = pred
    return out


def plot_norms(pipe, model_name="base", only_terms=False):
    if not hasattr(pipe, "norms"):
        raise ValueError("no norms in pipe saved")
    
    n = len(pipe.norms[0])

    if only_terms:
        for values, label in zip(pipe.norms[-2:], ["term_1", "term_2"]):
            plt.plot(list(range(1, n + 1)), values, label=label) 
    else:
        for values, label in zip(pipe.norms, ["uncond", "out1", "out2", "term_1", "term_2"]):
            plt.plot(list(range(1, n + 1)), values, label=label) 

    plt.title(f"{model_name} norms")
    plt.legend()
    plt.show()