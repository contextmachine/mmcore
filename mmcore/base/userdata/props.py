from mmcore.base.ecs.components import apply, component, todict


@component()
class Props:
    ...

def apply_props(props: Props, val: dict):
    apply(props, val)


def props_to_dict(props):
    return todict(props)
