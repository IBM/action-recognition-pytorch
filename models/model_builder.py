from . import s3d, i3d, s3d_resnet, i3d_resnet, resnet, inception_v1, s3d_resnet_tam


MODEL_TABLE = {
    's3d': s3d,
    'i3d': i3d,
    's3d_resnet': s3d_resnet,
    's3d_resnet_tam': s3d_resnet_tam,
    'i3d_resnet': i3d_resnet,
    'resnet': resnet,
    'inception_v1': inception_v1
}


def build_model(args, test_mode=False):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    model = MODEL_TABLE[args.backbone_net](**vars(args))
    network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net
    arch_name = "{dataset}-{modality}-{arch_name}".format(
        dataset=args.dataset, modality=args.modality, arch_name=network_name)
    arch_name += "-f{}".format(args.groups)

    # add setting info only in training
    if not test_mode:
        arch_name += "-{}{}-bs{}-e{}".format(args.lr_scheduler, "-syncbn" if args.sync_bn else "",
                                             args.batch_size, args.epochs)
    return model, arch_name
