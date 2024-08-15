from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

class ZeroClamper(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'a'):    # clamp the ir.pos.a >0  (due to a**0.5)
            w = module.a.data
            w.clamp_(min=1e-6)

@HOOKS.register_module()
class BaseIRHook(Hook):
    def __init__(self):
        super(BaseIRHook, self).__init__()

    def after_train_iter(self, runner):
        model = runner.model
        clamper = ZeroClamper()
        if is_module_wrapper(model):
            model = model.module
        model.apply(clamper)
