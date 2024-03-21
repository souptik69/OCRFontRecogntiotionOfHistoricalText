import torch


def compute_unpadded_width_after_conv(input_width_batch, conv_params):
    output = []

    for input_width in input_width_batch:
        for idx, c_p in enumerate(conv_params):
            if idx == 0:
                output_width = _compute_output_width(input_width, c_p[0], c_p[1], c_p[2])  # first conv
            else:
                output_width = _compute_output_width(output_width, c_p[0], c_p[1], c_p[2])  # later convs
        output.append(output_width)

    return torch.tensor(output)


def _compute_output_width(width_in, kernel, stride, padding=0, dilation=1):
    x = width_in + 2 * padding - dilation * (kernel - 1) - 1
    return torch.clamp(torch.true_divide(x, stride).int() + 1, min=1)

# def compute_unpadded_width_after_conv(input_width_batch: List[int]):
#     output = []
#
#     for input_width in input_width_batch:
#         output_width = _compute_output_width(input_width, 4, 2)  # first conv
#         output_width = _compute_output_width(output_width, 4, 1)  # second conv
#         output_width = _compute_output_width(output_width, 2, 2)  # first max pool
#         output_width = _compute_output_width(output_width, 3, 1)  # third conv
#         output_width = _compute_output_width(output_width, 2, 2)  # second max pool
#         if output_width == 0:
#             output.append(1)
#         else:
#             output.append(output_width)
#
#     return torch.tensor(output)
#
#
# def _compute_output_width(width_in, kernel, stride, padding=0, dilation=1) -> int:
#     return int(torch.true_divide((width_in + 2 * padding - dilation * (kernel - 1) - 1),stride) + 1)
