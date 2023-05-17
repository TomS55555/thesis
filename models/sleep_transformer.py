import torch
import torch.nn as nn
import constants
import pytorch_lightning as pl
import math
import torch.optim as optim


class SleepTransformer(pl.LightningModule):
    """
        The SleepTransformer takes as input batches of a sequence of 'outer' time-frequency images of the form [batch x outer x inner x feat] where 
        'inner x feat' is one time-frequency image. The output are logits over the labels as follows: [batch x outer x label]
        
    """

    def __init__(self, outer_dim, inner_dim, feat_dim, dim_feedforward, num_heads, num_layers):
        super().__init__()
        self.save_hyperparameters()

        inner_transformer = InnerTransformer(feat_dim=feat_dim,
                                             inner_dim=inner_dim,
                                             dim_feedforward=dim_feedforward,
                                             num_heads=num_heads,
                                             num_layers=num_layers)
        outer_transformer = OuterTransformer(feat_dim=feat_dim,
                                             outer_dim=outer_dim,
                                             dim_feedforward=dim_feedforward,
                                             num_heads=num_heads,
                                             num_layers=num_layers)

        classifier = Classifier(feat_dim=feat_dim,
                                hidden_dim=dim_feedforward)

        self.net = nn.Sequential(inner_transformer,
                                 outer_transformer,
                                 classifier)

        self.loss_module = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def common_step(self, batch, calculate_loss=False):
        inputs, labels = batch
        batch_dim, outer_dim, inner_dim, feat_dim = inputs.shape
        preds = self.net(inputs)


        labels_plus = labels.view(batch_dim*outer_dim,)
        preds_plus = preds.view(batch_dim*outer_dim, constants.N_CLASSES)
        loss = self.loss_module(preds_plus, labels_plus.long()) if calculate_loss else None

        acc = (preds_plus.argmax(dim=-1) == labels_plus).float().mean()

        return acc, loss


    def training_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, calculate_loss=True)

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        acc, loss = self.common_step(batch, calculate_loss=True)
        self.log('val_acc', acc)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        acc, _ = self.common_step(batch, calculate_loss=False)

        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1e-4,
            eps=1e-7,
            betas=(0.9, 0.999)
        )
        return optimizer


class OuterTransformer(nn.Module):
    """
        This module takes as input tensors of the form [batch x outer x feat] and outputs a tensors of
        the form [batch x outer x feat] that can be used as input to a classifier
    """

    def __init__(self, outer_dim, feat_dim, dim_feedforward, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.outer_position_encoding = PositionalEncoding(
            sequence_length=outer_dim,
            hidden_size=feat_dim
        )

    def forward(self, x):
        return self.transformer(self.outer_position_encoding(x))


class InnerTransformer(nn.Module):
    """
        This module takes as input tensors of the form [batch x outer x inner x feat] and outputs tensors of the form [batch x outer x feat]
        For the SleepTransformer, inner x feat is a time frequency image of a 1D EEG time-series (inner is time, feat is freq dimension)
    """

    def __init__(self, inner_dim, feat_dim, dim_feedforward, num_heads, num_layers, include_aggregrator=True):
        super().__init__()
        self.dim_feedforward = dim_feedforward  # Size of hidden dimension used in MLP within transformerencoder layer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=num_layers)
        self.aggregator = Aggregator(feat_dim=feat_dim)
        self.inner_position_encoding = PositionalEncoding(
            sequence_length=inner_dim,
            hidden_size=feat_dim
        )
        self.include_aggregrator = include_aggregrator

    def forward(self, x):
        batch_dim, outer_dim, inner_dim, feat_dim = x.shape
        batch_plus = x.view(batch_dim * outer_dim, inner_dim, feat_dim)  # reshape before putting through transformer
        batch_plus = self.inner_position_encoding(batch_plus)  # Add positional encoding
        transformed_plus = self.transformer(batch_plus)
        if self.include_aggregrator:
            aggregrated_plus = self.aggregator(transformed_plus)
            return aggregrated_plus.view(batch_dim, outer_dim, feat_dim)
        else:
            return transformed_plus.view(batch_dim, outer_dim, inner_dim, feat_dim)


class Aggregator(nn.Module):
    """
        This class performs the final attention to reduce the dimensionality after the transformer encoder layer from [b+ x inner x feat] to [b+ x feat]
        It does this by using attention which results in a weighted sum
    """

    def __init__(self, feat_dim, hidden_dim=None, unsqeeze=False):
        """
            hidden_dim is the hidden dimension used by attention
            feat_dim is the feature dimension
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim
        # self.Wa = torch.randn((hidden_dim, feat_dim), requires_grad=True)
        # self.ba = torch.zeros(hidden_dim, requires_grad=True)
        self.ae = nn.Parameter(torch.randn((hidden_dim, 1), requires_grad=True))
        self.linear = nn.Linear(in_features=feat_dim,
                                out_features=hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.unsqueeze = unsqeeze

    def forward(self, x):
        # b, l, f = x.shape
        # ats = self.tanh(x @ self.Wa.T + self.ba.repeat(l,1).unsqueeze(0).repeat(b,1,1))  # maybe this can faster
        ats = self.tanh(self.linear(x))
        alphas = self.softmax(ats @ self.ae)
        result = torch.bmm(x.transpose(1, 2), alphas).squeeze(2)
        if not self.unsqueeze:
            return result
        else:
            return result.unsqueeze(1)


class Classifier(nn.Module):
    """
        This module takes as input tensors of the form [batch x outer x feat] and outputs tensors of
        the form
    """

    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                in_features=feat_dim,
                out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=constants.N_CLASSES
            )
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, hidden_size):
        super().__init__()
        self.encoding = get_positional_encoding(sequence_length, hidden_size)
        self.encoding.requires_grad = False

    def forward(self, x):
        self.encoding = self.encoding.to(x.device)
        x = x + self.encoding
        return x


def get_positional_encoding(sequence_length, hidden_size):
    # create a matrix of shape (sequence_length, hidden_size)
    position = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    encoding = torch.cat([sin, cos], dim=1)
    return encoding
