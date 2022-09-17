from torch.utils.checkpoint import checkpoint


def checkpoint_factory(model):
    origin_forward = model.forward

    def checkpoint_hooker(*args, **kwargs):
        def checkpoint_forward(*args, **kwargs):
            return checkpoint(origin_forward, *args, **kwargs)
        try:
            assert any(p.requires_grad for p in args)
            res = checkpoint_forward(*args, **kwargs)
            model.forward = checkpoint_forward
        except Exception:
            res = origin_forward(*args, **kwargs)
            model.forward = origin_forward
        return res
    return checkpoint_hooker


def dfs_set_checkpoint(model):
    children = list(model.children())
    if len(children) == 0:
        model.forward = checkpoint_factory(model)
    else:
        [dfs_set_checkpoint(c) for c in children]


def prefix_state_dict(state_dict, prefix, replace=""):
    prefix, replace = prefix + ".", replace + "." if replace else ""
    n, res = len(prefix), {}
    for k in state_dict.keys():
        if k[:n] == prefix:
            res[replace + k[n:]] = state_dict[k]
    return res
