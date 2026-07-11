import torch

from Model.Sun_Net_gan import Discriminator
from models.generator import build_generator


def main():
    torch.manual_seed(123)
    x_opt = torch.rand(1, 3, 64, 64) * 2.0 - 1.0
    x_sar = torch.rand(1, 3, 64, 64) * 2.0 - 1.0

    generator = build_generator(input_nc=3, output_nc=3, n_residual_blocks=1).eval()
    cd_model = Discriminator(
        input_shape=(3, 64, 64),
        use_cgdr=True,
        cgdr_max_flow=6.0,
        cgdr_corr_threshold=0.45,
        cgdr_region_mode="correlation",
    ).eval()

    with torch.no_grad():
        generated = generator(x_opt)
        _, _, prediction, reg_loss = cd_model(
            x_sar,
            generated,
            return_registration=True,
        )

    assert generated.shape == x_opt.shape
    assert prediction.shape[:2] == (1, 2)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(reg_loss)
    print("AEF-GLCD smoke test passed.")
    print(f"Generator output: {tuple(generated.shape)}")
    print(f"CD output: {tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
