"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import Linear

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class MyDecoder(Decoder):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c, args):
        super(MyDecoder, self).__init__(c)
        self.input_dim = args.num_centroid
        self.output_dim = args.n_classes
        act = lambda x: x
        self.cls = Linear(args, self.input_dim, self.output_dim, 0.0, act, args.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = x
        return super(MyDecoder, self).decode(h, adj)

model2decoder = {
    'H2HGCN': MyDecoder,
}

