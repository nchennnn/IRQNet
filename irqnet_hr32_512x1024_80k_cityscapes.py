_base_ = [
    './hr18_ir.py', '../cityscapes.py',
    '../default_runtime.py', '../schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(32, 64)),
            stage3=dict(num_channels=(32, 64, 128)),
            stage4=dict(num_channels=(32, 64, 128, 256)))),
    decode_head=dict(
        wpe=True,
        df=True,
        featfocus=True,
        pos_dim=48,
        num_classes=19,
        in_channels=[32, 64, 128, 256],
        channels=256
    )
)
custom_hooks = [dict(type="BaseIRHook")]

# fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
# fp16 = dict()