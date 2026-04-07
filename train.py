from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(
        opt.feature_type,
        fcs_loss_weight=opt.fcs_loss_weight,
        fcs_predictor_path=opt.fcs_predictor_path,
        com_loss_weight=opt.com_loss_weight,
        bilateral_loss_weight=opt.bilateral_loss_weight,
        foot_height_loss_weight=opt.foot_height_loss_weight,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
