def create_model(opt, pretrained=None):
    from .HG_model import HGModel
    model = HGModel(opt, pretrained)
    return model
