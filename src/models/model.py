import numpy as np
from torch import all, any, int64, nn, no_grad, ones, sum, tensor, topk, zeros
from torch.nn.functional import softmax

from src.models.components.cnn import CNN
from src.models.components.swin import SwinTransformer
from src.models.components.transformer import Transformer


class Net(nn.Module):
    def __init__(self, backbone, transformer):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer

    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (B, C, H, W)
            - tgt_input: (T, B)
            - tgt_key_padding_mask: (B, T)
            -> output: (B, T, V)
        """

        src = self.backbone(img)

        return self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)

    def predict(self, img, max_seq_length=128):
        self.eval()

        device = img.device
        batch_len = len(img)

        with no_grad():
            src = self.backbone(img)
            memory = self.transformer.forward_encoder(src)

            translated_sent = [[1] * batch_len]
            prob = [[1] * batch_len]

            sent_length = 0

            while sent_length <= max_seq_length and not all(
                any(tensor(translated_sent).T == 2, dim=1)
            ):
                tgt_inp = tensor(translated_sent, dtype=int64, device=device)
                output, memory = self.transformer.forward_decoder(tgt_inp, memory)
                output = softmax(output, dim=-1)
                output = output.cpu()

                values, indices = topk(output, 5)

                prob.append(values[:, -1, 0])
                translated_sent.append(indices[:, -1, 0])

                sent_length += 1

                del output

            translated_sent = tensor(translated_sent).T

            prob = tensor(prob).T
            prob = prob * (translated_sent > 3)
            prob = sum(prob, -1) / sum(prob > 0, -1)

        return translated_sent, prob
