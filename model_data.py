import torch
import torch.nn as nn
from torch.autograd import Variable


class CNNClassificationModel(nn.Module):
    def __init__(self,
                 use_pretrained_vector=False,
                 word_count=300000,
                 embedding_size=128,
                 number_of_classes=2,
                 batch_size=50,
                 keep_embeddings_static=False,
                 pretrained_vector_weight=None,
                 use_multi_channel=False):
        super(CNNClassificationModel, self).__init__()

        self.keep_embeddings_static = keep_embeddings_static
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.input_channel = 1
        self.use_multi_channel = use_multi_channel

        # Setting up embeddings
        if use_pretrained_vector:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_vector_weight),
                                                                freeze=False)
            self.embedding_size = pretrained_vector_weight.shape[1]
        else:
            self.embedding_layer = nn.Embedding(word_count, embedding_size)
            self.embedding_size = embedding_size
            nn.init.uniform_(self.embedding_layer.weight, -1.0, 1.0)

        if use_multi_channel:
            self.embedding_layer2 = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_vector_weight),
                                                                 freeze=True)
            self.input_channel = 2


        self.convolution_layer_3dfilter = nn.Conv2d(self.input_channel, 100, (3, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_3dfilter.weight)

        self.convolution_layer_4dfilter = nn.Conv1d(self.input_channel, 100, (4, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_4dfilter.weight)

        self.convolution_layer_5dfilter = nn.Conv1d(self.input_channel, 100, (5, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_4dfilter.weight)

        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(300, self.number_of_classes)
        nn.init.xavier_uniform_(self.linear.weight)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        embedded = self.embedding_layer(input)
        if self.keep_embeddings_static:
            embedded = Variable(embedded)

        if self.use_multi_channel:
            embedded2 = self.embedding_layer2(input)
            embedded = torch.stack([embedded, embedded2], dim=1)
        else:
            embedded = embedded.unsqueeze(1)

        conv_opt3 = self.convolution_layer_3dfilter(embedded)
        conv_opt4 = self.convolution_layer_4dfilter(embedded)
        conv_opt5 = self.convolution_layer_5dfilter(embedded)

        conv_opt3 = nn.functional.relu(conv_opt3).squeeze(3)
        conv_opt4 = nn.functional.relu(conv_opt4).squeeze(3)
        conv_opt5 = nn.functional.relu(conv_opt5).squeeze(3)

        conv_opt3 = nn.functional.max_pool1d(conv_opt3, conv_opt3.size(2)).squeeze(2)
        conv_opt4 = nn.functional.max_pool1d(conv_opt4, conv_opt4.size(2)).squeeze(2)
        conv_opt5 = nn.functional.max_pool1d(conv_opt5, conv_opt5.size(2)).squeeze(2)


        conv_opt = torch.cat((conv_opt3, conv_opt4, conv_opt5), 1)
        conv_opt = self.dropout(conv_opt)

        linear_opt = self.linear(conv_opt)

        return linear_opt

    ## I used sentences with Variable  earlier as implemented in the paper but as the training was too slow
    ## sentences were padded to maximum sentence length
    # def forward(self, input):
    #     conv_output = []
    #     for inp in input:
    #         embedded = self.embedding_layer(inp).view(1, self.embedding_size, -1)
    #         if self.keep_embeddings_static:
    #             embedded = Variable(embedded)
    #         conv_opt3 = self.convolution_layer_3dfilter(embedded)
    #         conv_opt4 = self.convolution_layer_4dfilter(embedded)
    #         conv_opt5 = self.convolution_layer_5dfilter(embedded)
    #         conv_opt3 = nn.functional.relu(conv_opt3)
    #         conv_opt4 = nn.functional.relu(conv_opt4)
    #         conv_opt5 = nn.functional.relu(conv_opt5)
    #
    #         # Maxpooling to take out the max from each one 100 fitera
    #         conv_opt3 = nn.functional.max_pool1d(conv_opt3, conv_opt3.size(2))
    #         conv_opt4 = nn.functional.max_pool1d(conv_opt4, conv_opt4.size(2))
    #         conv_opt5 = nn.functional.max_pool1d(conv_opt5, conv_opt5.size(2))
    #
    #         conv_opt = torch.cat((conv_opt3, conv_opt4, conv_opt5), 2).view(1, -1)
    #         conv_output.append(conv_opt)
    #
    #     conv_output = torch.cat(conv_output, 0)
    #     conv_output = self.dropout(conv_output)
    #
    #     output = self.linear(conv_output)
    #
    #     return output