from math import cos, pi

T=1000
W_SCALES=[1., 1., 1., 1.]

default_sch_kwargs = {
    "a": dict(),
    "b": dict()
}

class CustomScheduler:
    def __init__(self, w_values):
        self.w_values = w_values

    def __call__(self, w, i, *args, **kwargs):
        return self.w_values[i] # i - index of timestep (!!!)


def default_scheduler(w, t, *args, **kwargs):
    return w

def linear_scheduler(w, t, *args, **kwargs):
    return w * (1 - t/T)

def inv_linear_scheduler(w, t, *args, **kwargs):
    return w * (t/T)

def clamp_linear_scheduler(w, t, c, *args, **kwargs):
    return max(c, linear_scheduler(w, t))

def inv_clamp_linear_scheduler(w, t, c, *args, **kwargs):
    return max(c, inv_linear_scheduler(w, t))

def cosine_scheduler(w, t, *args, **kwargs):
    return w * (cos(pi*t/T) + 1)

def inv_cosine_scheduler(w, t, *args, **kwargs):
    return cosine_scheduler(w, T - t, *args, **kwargs)

def power_cosine_scheduler(w, t, s, *args, **kwargs):
    return 0.5 * w * (1 - cos(pi * ((T - t) / T)**s))

def inv_power_cosine_scheduler(w, t, s, *args, **kwargs):
    return power_cosine_scheduler(w, T - t, s, *args, **kwargs)

def clamped_pcs_scheduler(w, t, c, s, *args, **kwargs):
    return max(c, power_cosine_scheduler(w, t, s))

def inv_clamped_pcs_scheduler(w, t, c, s, *args, **kwargs):
    return clamped_pcs_scheduler(w, T - t, c, s, *args, **kwargs)

def cap_shape_scheduler(w, t, *args, **kwargs):
    if t > T / 2:
        return linear_scheduler(w, t)
    return w * t/T

def ccap(w, t, c, *args, **kwargs):
    return max(c, cap_shape_scheduler(w, t))

name2sch = {
    "default": default_scheduler,
    "linear": linear_scheduler,
    "clamped_lin": clamp_linear_scheduler,
    "cos": cosine_scheduler,
    "pcs": power_cosine_scheduler,
    "clamped_pcs": clamped_pcs_scheduler,
    "capshape": cap_shape_scheduler,
    "ccshape": ccap,
    "custom": CustomScheduler,
    "inv_cos": inv_cosine_scheduler,
    "inv_pcs": inv_power_cosine_scheduler,
    "inv_clamped_pcs": inv_clamped_pcs_scheduler,
    "inv_linear": inv_linear_scheduler,
    "inv_clamp_linear": inv_clamp_linear_scheduler
}
def get_scheduler(sch_name: str):
    return name2sch[sch_name]