import torch


class DiffusionSchedule:
    def __init__(
        self,
        timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
    ):
        self.timesteps = timesteps

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=device,
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0, noise, t):
        alpha_bar = self.alpha_bars[t]
        alpha_bar = alpha_bar[:, None, None, None]

        return (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        condition,
        device,
    ):
        model.eval()

        x = torch.randn_like(condition)

        for i in reversed(range(self.timesteps)):

            t = torch.full(
                (condition.shape[0],),
                i,
                device=device,
                dtype=torch.long,
            )

            predicted_noise = model(
                x,
                condition,
                t,
            )

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                (1.0 / torch.sqrt(alpha))
                * (
                    x
                    - (
                        beta
                        / torch.sqrt(1.0 - alpha_bar)
                    )
                    * predicted_noise
                )
                + torch.sqrt(beta) * noise
            )

        return x.clamp(0.0, 1.0)