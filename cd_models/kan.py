import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.io import DataLoader, IterableDataset

class KANLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=paddle.nn.Silu,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            paddle.arange(-spline_order, grid_size + spline_order + 1, dtype=paddle.float32) * h
            + grid_range[0]
        ).expand([in_features, -1]).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = self.create_parameter(
            shape=[out_features, in_features], default_initializer=paddle.nn.initializer.Constant(value=scale_base))
        self.spline_weight = self.create_parameter(
            shape=[out_features, in_features, grid_size + spline_order], default_initializer=paddle.nn.initializer.Constant(value=scale_spline))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        self.base_weight.set_value(paddle.full([self.out_features, self.in_features], self.scale_base))
        with paddle.no_grad():
            noise = (
                paddle.rand([self.grid_size + 1, self.in_features, self.out_features], dtype=paddle.float32)
                - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.set_value(
                self.scale_spline
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order],
                    noise,
                )
            )

    def b_splines(self, x: paddle.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            paddle.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features

        grid: paddle.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).cast(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert tuple(bases.shape) == tuple((
            x.shape[0],
            self.in_features,
            self.grid_size + self.spline_order,
        ))
        return bases


    def curve2coeff(self, x: paddle.Tensor, y: paddle.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).
            y (paddle.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            paddle.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features
        assert y.shape == [x.shape[0], self.in_features, self.out_features]

        A = self.b_splines(x).transpose(
            [1, 0, 2]
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose([1, 0, 2])  # (in_features, batch_size, out_features)
        solution = paddle.linalg.lstsq(
            A, B
        )  # solution: (in_features, grid_size + spline_order, out_features)

        result = solution[0].transpose(
            [2, 0, 1]
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.shape == [
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        ]
        return result
    
    def forward(self, x: paddle.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight.transpose([1, 0]))
        spline_output = F.linear(
            self.b_splines(x).reshape([x.shape[0], -1]),
            self.spline_weight.reshape([self.out_features, -1]).transpose([1, 0])
        )
        return base_output + spline_output
    
    @paddle.no_grad()
    def update_grid(self, x: paddle.Tensor, margin=0.01):
        assert x.ndim == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.transpose([1, 0, 2])  # (in, batch, coeff)
        orig_coeff = self.spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.transpose([1, 2, 0])  # (in, coeff, out)
        unreduced_spline_output = paddle.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.transpose([1, 0, 2])  # (batch, in, out)
       
        # Sort each channel individually to collect data distribution
        x_sorted = paddle.sort(x, axis=0)
        grid_adaptive = x_sorted[
            paddle.linspace(
                0, batch - 1, self.grid_size + 1, dtype='int64'
            ).astype('int32')
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            paddle.arange(
                self.grid_size + 1, dtype='float32'
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = paddle.concat(
            [
                grid[:1]
                - uniform_step
                * paddle.arange(self.spline_order, 0, -1, dtype='float32').unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * paddle.arange(1, self.spline_order + 1, dtype='float32').unsqueeze(1),
            ],
            axis=0
        )

        self.grid.set_value(grid.T)
        self.spline_weight.set_value(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_norm = paddle.mean(paddle.abs(self.spline_weight), axis=-1)
        reg_loss_activation = paddle.sum(l1_norm)
        p = l1_norm / reg_loss_activation
        reg_loss_entropy = -paddle.sum(p * paddle.log(p + 1e-8))
        return (
            regularize_activation * reg_loss_activation +
            regularize_entropy * reg_loss_entropy
        )

class KAN(paddle.nn.Layer):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=paddle.nn.Silu,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.layers = paddle.nn.LayerList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: paddle.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class RandomDataGenerator(IterableDataset):
    def __init__(self, batch_size, num_samples):
        self.batch_size = batch_size
        self.num_samples = num_samples
    
    def __iter__(self):
        for _ in range(self.num_samples):
            x = paddle.rand([self.batch_size, 2])
            u = x[:, 0]
            v = x[:, 1]
            y = (u + v) / (1 + u * v)
            y = y.unsqueeze(-1)
            yield x, y

def test_mul():
    kan = KAN([2, 2, 1], base_activation=nn.Identity)
    optimizer = paddle.optimizer.LBFGS(parameters=kan.parameters(), learning_rate=1)
    dataloader = DataLoader(RandomDataGenerator(1024, 1000), batch_size=None)

    for i, (x, y) in enumerate(dataloader):
        def closure():
            pred_y = kan(x, update_grid=(i % 20 == 0))
            loss = paddle.nn.functional.mse_loss(pred_y.squeeze(-1), y.squeeze(-1))
            reg_loss = kan.regularization_loss(1, 0)
            total_loss = loss + 1e-5 * reg_loss
            print(f"Iteration {i}: MSE Loss = {loss.numpy()}, Regularization Loss = {reg_loss.numpy()}")
            return total_loss
        
        optimizer.step(closure)
        optimizer.clear_grad()

    for layer in kan.layers:
        print(layer.spline_weight)

if __name__ == "__main__":
    test_mul()
