import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import register


def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


@register('gaussian-splatter')
class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""
    def __init__(self, kernel_size, num_row_points=48, num_column_points=48):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            num_row_points (int): The number of points in the row dimension of the Gaussian grid.
            num_column_points (int): The number of points in the column dimension of the Gaussian grid.
        """
        super(GaussianSplatter, self).__init__()

        # key parameter in 2D Gaussian Splatter
        self.kernel_size = kernel_size
        self.row = num_row_points
        self.column = num_column_points
        self.num_points = num_column_points * num_row_points

        # Initialization
        sigma_values = torch.rand(self.row * self.column, 2, requires_grad=True)
        self.sigma_x, self.sigma_y = sigma_values[:, 0], sigma_values[:, 1]
        self.opacity = torch.ones(self.row * self.column, 1, requires_grad=True)

        # 3 trainable parameters: sigma_x, sigma_y, opacity
        self.sigma_x = nn.Parameter(self.sigma_x)  # std in x-axis
        self.sigma_y = nn.Parameter(self.sigma_y)  # std in y-axis
        self.opacity = nn.Parameter(self.opacity)  # transparency of feature, shape=[num_points, 1], each element in batch share the same opacity

    def forward(self, inp):
        """
        Apply 2D Gaussian splatting to the input features.
        Args:
            inp (torch.Tensor): The input features of shape [Batch, C, H, W] from encoder.
        Returns:
            torch.Tensor: The output features after Gaussian splatting, of the same shape as the input.
        """
        device = inp.device
        image_size = inp.shape

        # Unfold the input to many small patches to avoid extreme GPU memory consumption
        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(inp)
        # Unfolded_feature dimension becomes [Batch, C*K*K, L], where L is the number of columns after unfolding
        L = unfolded_feature.shape[-1]
        unfolded_feature_reshaped = unfolded_feature.transpose(1, 2).reshape(image_size[0] * L, image_size[1],
                                                                             self.row, self.column)

        inp = unfolded_feature_reshaped
        coords_ = generate_meshgrid(inp.shape[-2], inp.shape[-1])
        num_feature_points = inp.shape[-2] * inp.shape[-1]
        colors_, coords_norm = fetching_features_from_tensor(inp, coords_)

        """Replicates and rearranges feature attributes of 2d gaussian grid to fit the input spatial domain
           at arbitrary scales and maintaining the original pattern by indexing using modular arithmetic."""
        new_row, new_column = inp.shape[-2], inp.shape[-1]

        # Generate expanded row/column indices for the new domain by wrapping around the original number of rows/columns
        expanded_row_indices = torch.arange(new_row) % self.row
        expanded_column_indices = torch.arange(new_column) % self.column

        # Compute a 2D grid of indices that map the new domain back to the original domain
        expanded_indices = expanded_row_indices.unsqueeze(1) * self.column + expanded_column_indices
        expanded_indices_flat = expanded_indices.view(-1)

        expanded_sigma_x = self.sigma_x[expanded_indices_flat]
        expanded_sigma_y = self.sigma_y[expanded_indices_flat]
        expanded_opacity = self.opacity[expanded_indices_flat]

        # Spread Gaussian points over the whole feature map
        batch_size, channel, _, _ = inp.shape
        sigma_x = expanded_sigma_x.view(num_feature_points, 1, 1)
        sigma_y = expanded_sigma_y.view(num_feature_points, 1, 1)
        covariance = torch.stack(
            [torch.stack([sigma_x ** 2, torch.zeros(sigma_x.shape, device=device)], dim=-1),
             torch.stack([torch.zeros(sigma_x.shape, device=device), sigma_y ** 2], dim=-1)], dim=-2
        )  # when correlation rou is set to zero, covariance will always be positive semi-definite
        inv_covariance = torch.inverse(covariance).to(device)

        # Choosing a broad range for the distribution [-5,5] to avoid any clipping
        start = torch.tensor([-5.0], device=device).view(-1, 1)
        end = torch.tensor([5.0], device=device).view(-1, 1)
        base_linspace = torch.linspace(0, 1, steps=self.kernel_size, device=device)
        ax_batch = start + (end - start) * base_linspace

        # Expanding dims for broadcasting
        ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, self.kernel_size, -1)

        # Creating a batch-wise meshgrid using broadcasting
        xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

        xy = torch.stack([xx, yy], dim=-1)
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
        kernel = torch.exp(z) / (
                2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance)).to(device).view(num_feature_points, 1, 1))

        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
        kernel_normalized = kernel / kernel_max_2

        kernel_reshaped = kernel_normalized.repeat(1, channel, 1).view(num_feature_points * channel, self.kernel_size,
                                                                       self.kernel_size)
        kernel_color = kernel_reshaped.unsqueeze(0).reshape(num_feature_points, channel, self.kernel_size, self.kernel_size)

        # Calculating the padding needed to match the image size
        pad_h = self.row - self.kernel_size
        pad_w = self.column - self.kernel_size

        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")

        # Adding padding to make kernel size equal to the image size
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)

        kernel_color_padded = torch.nn.functional.pad(kernel_color, padding, "constant", 0)

        # Extracting shape information
        b, c, h, w = kernel_color_padded.shape

        # Create a batch of 2D affine matrices
        theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = coords_norm

        # Creating grid and performing grid sampling
        grid = F.affine_grid(theta, size=[b, c, h, w], align_corners=True)
        kernel_color_padded_translated = F.grid_sample(kernel_color_padded, grid, align_corners=True)
        kernel_color_padded_translated = kernel_color_padded_translated.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        colors = colors_ * expanded_opacity.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        color_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)
        final_image_layers = color_values_reshaped * kernel_color_padded_translated
        final_image = final_image_layers.sum(dim=1)
        final_image = torch.clamp(final_image, 0, 1)

        # Fold the input back to the original size
        fold = nn.Fold(output_size=(image_size[-2], image_size[-1]), kernel_size=(self.row, self.column),
                       stride=(self.row, self.column))
        final_image = final_image.reshape(image_size[0], L, image_size[1] * self.row * self.column).transpose(1, 2)
        final_image = fold(final_image)

        return final_image


@register('latent-gaussian')
class Latent_Gaussian(nn.Module):
    """A module that applies latent 2D Gaussian splatting to input features. The input is projected to a latent
    space and then passes to the gaussian module, and finally to an expand net to recover the original shape"""
    def __init__(self, channel, kernel_size, squeeze_scale):
        """
        Initialize the latent 2D Gaussian Splatter module. Input (of shape [batch, channel, height, width]) is
        first downscaled to [batch, channel//scale, height//scale, width//scale] and apply gaussian splatting on
        a compact dimension to reduce GPU memory consumption. Original shape is then restored by an expansion net.
        Args:
            channel (int): The channel of the input generated by the encoder.
            kernel_size (int): The size of the kernel to convert rasterization.
            squeeze_scale (int): The scale factor for the squeeze and expand operation.
        """
        super(Latent_Gaussian, self).__init__()

        # key parameter in Latent Gaussian Splatter
        self.kernel_size = kernel_size
        # Compress the input tensor to [batch, channel//scale, height//scale, width//scale]
        self.squeeze_scale = round(squeeze_scale)
        self.squeeze_net = nn.Conv2d(channel, channel//squeeze_scale, kernel_size=squeeze_scale, stride=squeeze_scale)
        self.expand_net = nn.ConvTranspose2d(channel//squeeze_scale, channel, kernel_size=squeeze_scale, stride=squeeze_scale)
        self.gaussian = GaussianSplatter(kernel_size=kernel_size, num_row_points=48//squeeze_scale,
                                         num_column_points=48//squeeze_scale)

    def forward(self, inp):
        """
        Apply latent 2D Gaussian splatting to the input features.
        Args:
            inp (torch.Tensor): The input features of shape [Batch, C, H, W] from encoder.
        Returns:
            torch.Tensor: The output features after Latent Gaussian splatting, of the same shape as the input.
        """
        latent_input = self.squeeze_net(inp)
        compact_gaussian = self.gaussian(latent_input)
        output = self.expand_net(compact_gaussian, output_size=inp.shape[-2:])
        return output + inp
