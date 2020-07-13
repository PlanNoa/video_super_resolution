from .options.train_options import TrainOptions
from .models.models import create_model

__all__ = ['HourGlass']


def HourGlass(pretrained=None):
    opt = TrainOptions().parse()
    model = create_model(opt, pretrained)
    return model.netG
