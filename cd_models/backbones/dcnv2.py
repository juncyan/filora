# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np
from paddle.autograd import PyLayer


class DCN_V2Layer(nn.Layer):
    def __init__(self, sparse_feature_number=1100001,
                 sparse_feature_dim=40,#10,40
                 dense_feature_dim=13,
                 sparse_num_field=26,
                 layer_sizes=[768, 768], # [768, 768]  [500, 500, 500],
                 cross_num=2,
                 is_Stacked=True,
                 use_low_rank_mixture=True,
                 low_rank=256,
                 num_experts=4):
        super(DCN_V2Layer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.num_field = sparse_num_field + dense_feature_dim
        self.layer_sizes = layer_sizes
        self.cross_num = cross_num
        self.is_Stacked = is_Stacked
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts

        self.init_value_ = 0.1

        use_sparse = True
        if paddle.is_compiled_with_custom_device('npu'):
            use_sparse = False

        # sparse coding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=use_sparse,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.dense_emb = nn.Linear(self.dense_feature_dim, (
            self.sparse_feature_dim * self.dense_feature_dim))

        self.DeepCrossLayer_ = DeepCrossLayer(
            sparse_num_field, sparse_feature_dim, dense_feature_dim, cross_num,
            use_low_rank_mixture, low_rank, num_experts)

        self.DNN_ = DNNLayer(
            sparse_feature_dim,
            dense_feature_dim,
            sparse_num_field,
            layer_sizes,
            dropout_rate=0.5)

        if self.is_Stacked:
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1],
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(self.layer_sizes[-1]))))

        else:
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1] +
                (dense_feature_dim + sparse_num_field
                 ) * self.sparse_feature_dim,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(self.layer_sizes[
                            -1] + dense_feature_dim * sparse_num_field))))

    def forward(self, sparse_inputs, dense_inputs):
        # print("sparse_inputs:",sparse_inputs)
        # print("dense_inputs:",dense_inputs)
        # EmbeddingLayer
        sparse_inputs_concat = paddle.concat(
            sparse_inputs, axis=1)  #Tensor(shape=[bs, 26])
        sparse_embeddings = self.embedding(
            sparse_inputs_concat)  # shape=[bs, 26, dim]

        # print("sparse_embeddings shape:",sparse_embeddings.shape)

        sparse_embeddings_re = paddle.reshape(
            sparse_embeddings,
            shape=[-1, self.sparse_num_field * self.sparse_feature_dim])

        dense_embeddings = self.dense_emb(
            dense_inputs)  # # shape=[bs, 13, dim]

        feat_embeddings = paddle.concat(
            [sparse_embeddings_re, dense_embeddings], 1)
        # print("feat_embeddings:",feat_embeddings.shape)

        # Model Structaul: Stacked or Parallel
        if self.is_Stacked:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(feat_embeddings)
            # MLPLayer
            dnn_output = self.DNN_(cross_out)

            # print('----dnn_output shape----',dnn_output.shape)

            logit = self.fc(dnn_output)
            predict = F.sigmoid(logit)

        else:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(feat_embeddings)

            # MLPLayer
            dnn_output = self.DNN_(feat_embeddings)

            last_out = paddle.concat([dnn_output, cross_out], axis=-1)

            # print('----last_out_output shape----',last_out.shape)

            logit = self.fc(last_out)
            predict = F.sigmoid(logit)

        return predict


class DNNLayer(paddle.nn.Layer):
    def __init__(self,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 layer_sizes,
                 dropout_rate=0.5):
        super(DNNLayer, self).__init__()

        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = dense_feature_dim + sparse_num_field
        self.layer_sizes = layer_sizes
        self.sparse_num_field = sparse_num_field

        self.input_size = int((self.sparse_num_field + self.dense_feature_dim)
                              * self.sparse_feature_dim)

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

        sizes = [self.input_size] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    regularizer=paddle.regularizer.L2Decay(1e-7),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, feat_embeddings):
        # y_dnn = paddle.reshape(feat_embeddings,[feat_embeddings.shape[0], -1])
        y_dnn = feat_embeddings
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            y_dnn = self.drop_out(y_dnn)
        return y_dnn


class DeepCrossLayer(nn.Layer):
    def __init__(self, sparse_num_field, sparse_feature_dim, dense_feature_dim,
                 cross_num, use_low_rank_mixture, low_rank, num_experts):
        super(DeepCrossLayer, self).__init__()

        self.use_low_rank_mixture = use_low_rank_mixture
        self.input_dim = (
            sparse_num_field + dense_feature_dim) * sparse_feature_dim
        self.num_experts = num_experts
        self.low_rank = low_rank
        self.cross_num = cross_num

        if self.use_low_rank_mixture:
            self.crossNet = CrossNetMix(
                self.input_dim,
                layer_num=self.cross_num,
                low_rank=self.low_rank,
                num_experts=self.num_experts)
        else:
            self.crossNet = CrossNetV2(self.input_dim, self.cross_num)

    def forward(self, feat_embeddings):
        outputs = self.crossNet(feat_embeddings)

        return outputs


class CrossNetV2(nn.Layer):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()

        self.num_layers = num_layers
        self.cross_layers = nn.LayerList(
            nn.Linear(input_dim, input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


class CrossNetMix(nn.Layer):
    """ CrossNetMix improves CrossNet by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
    """

    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, in_features, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        # V: (in_features, low_rank)
        self.V_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, in_features, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        # C: (low_rank, low_rank)
        self.C_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, low_rank, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        self.gating = nn.LayerList(
            [nn.Linear(in_features, 1) for i in range(self.num_experts)])

        self.bias = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[in_features, 1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=0.0))
            for i in range(self.layer_num)
        ])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](
                    x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = paddle.matmul(self.V_list[i][expert_id].t(),
                                    x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = paddle.tanh(v_x)
                v_x = paddle.matmul(self.C_list[i][expert_id], v_x)
                v_x = paddle.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = paddle.matmul(self.U_list[i][expert_id],
                                     v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = paddle.stack(
                output_of_experts, axis=2)  # (bs, in_features, num_experts)
            gating_score_of_experts = paddle.stack(
                gating_score_of_experts, axis=1)  # (bs, num_experts, 1)
            moe_out = paddle.matmul(
                output_of_experts, F.softmax(
                    gating_score_of_experts, axis=1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class DygraphModel():

    # define model
    def create_model(self, config):
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        fc_sizes = config.get("hyper_parameters.fc_sizes")
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')
        cross_num = config.get("hyper_parameters.cross_num")
        l2_reg_cross = config.get("hyper_parameters.l2_reg_cross", None)
        clip_by_norm = config.get("hyper_parameters.clip_by_norm", None)
        is_Stacked = config.get("hyper_parameters.is_Stacked", None)
        use_low_rank_mixture = config.get(
            "hyper_parameters.use_low_rank_mixture", None)
        low_rank = config.get("hyper_parameters.low_rank", 32)
        num_experts = config.get("hyper_parameters.num_experts", 4)
        dnn_use_bn = config.get("hyper_parameters.dnn_use_bn", None)

        dcn_v2_model = DCN_V2Layer(
            sparse_feature_number, sparse_feature_dim, dense_feature_dim,
            sparse_input_slot - 1, fc_sizes, cross_num, is_Stacked,
            use_low_rank_mixture, low_rank, num_experts)

        return dcn_v2_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        # print("----batch_data", batch_data[0])
        # print("----batch_data", batch_data[1])
        # print("----batch_data", batch_data[-1])
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []

        for b in batch_data[:-1]:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
        dense_tensor = paddle.to_tensor(batch_data[-1].numpy().astype(
            'float32').reshape(-1, dense_feature_dim))
        label = sparse_tensor[0]

        # print("-----dygraph-----label:----",label.shape)
        # print("-----dygraph-----sparse_tensor[1:]:----", sparse_tensor[1:])
        # print("-----dygraph-----dense_tensor:----", dense_tensor)
        return label, sparse_tensor[1:], dense_tensor

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        # print("---dygraph----pred, label:",pred, label)
        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        # add l2_loss.............
        # print("---dygraph-----cost,avg_cost----",cost,avg_cost)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        clip_by_norm = config.get("hyper_parameters.optimizer.clip_by_norm",
                                  10.0)
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_by_norm)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters(), grad_clip=clip)
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)
        # print("---dygraph-----label, sparse_tensor, dense_tensor",label, sparse_tensor, dense_tensor)
        pred = dy_model.forward(sparse_tensor, dense_tensor)

        log_loss = self.create_loss(pred, label)

        # l2_reg_cross = config.get("hyper_parameters.l2_reg_cross", None)

        # for param in dy_model.DeepCrossLayer_.W.parameters():
        #     log_loss += l2_reg_cross * paddle.norm(param, p=2)

        # loss = log_loss + l2_loss
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---dygraph----pred,loss,predict_2d---",pred,loss,predict_2d)
        # print("---dygraph----metrics_list",metrics_list)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss}
        print_dict = {'log_loss': log_loss}
        # print_dict = None
        return log_loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)
        # print("----label, sparse_tensor, dense_tensor",label, sparse_tensor, dense_tensor)
        pred = dy_model.forward(sparse_tensor, dense_tensor)
        # update metrics
        log_loss = self.create_loss(pred, label)
        print_dict = {'log_loss': log_loss}

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---pred,predict_2d---",pred,predict_2d)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        # print("---metrics_list",metrics_list)
        return metrics_list, print_dict
        # return metrics_list, None



class DCNv2(nn.Layer):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        新增modulation 参数： 是DCNv2中引入的调制标量
        """
        super(DCNv2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2D(padding)
        self.conv = nn.Conv2D(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias_attr=bias)

        self.p_conv = nn.Conv2D(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1,
                                stride=stride,weight_attr=paddle.ParamAttr(initializer=0.))
        # 输出通道是2N

        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:  # 如果需要进行调制
            # 输出通道是N
            self.m_conv = nn.Conv2D(inc, kernel_size * kernel_size, kernel_size=3, padding=1,
                                    stride=stride,weight_attr=paddle.ParamAttr(initializer=0.))
            self.m_conv.register_backward_hook(self._set_lr)  # 在指定网络层执行完backward（）之后调用钩子函数

    @staticmethod
    def _set_lr(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):  # x: (b,c,h,w)
        offset = self.p_conv(x)  # (b,2N,h,w) 学习到的偏移量 2N表示在x轴方向的偏移和在y轴方向的偏移
        if self.modulation:  # 如果需要调制
            m = F.sigmoid(self.m_conv(x))  # (b,N,h,w) 学习到的N个调制标量

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = paddle.concat([paddle.clip(q_lt[..., :N], 0, x.size(2) - 1), paddle.clip(q_lt[..., N:], 0, x.size(3) - 1)],
                         axis=-1).long()
        q_rb = paddle.concat([paddle.clip(q_rb[..., :N], 0, x.size(2) - 1), paddle.clip(q_rb[..., N:], 0, x.size(3) - 1)],
                         axis=-1).long()
        q_lb = paddle.concat([q_lt[..., :N], q_rb[..., N:]], axis=-1)
        q_rt = paddle.concat([q_rb[..., :N], q_lt[..., N:]], axis=-1)

        # clip p
        p = paddle.concat([paddle.clip(p[..., :N], 0, x.size(2) - 1), paddle.clip(p[..., N:], 0, x.size(3) - 1)], axis=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 如果需要调制
        if self.modulation:  # m: (b,N,h,w)
            m = m.contiguous().permute(0, 2, 3, 1)  # (b,h,w,N)
            m = m.unsqueeze(dim=1)  # (b,1,h,w,N)
            m = paddle.concat([m for _ in range(x_offset.size(1))], axis=1)  # (b,c,h,w,N)
            x_offset *= m  # 为偏移添加调制标量

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = paddle.meshgrid(
            paddle.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            paddle.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = paddle.concat([paddle.flatten(p_n_x), paddle.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = paddle.meshgrid(
            paddle.arange(1, h * self.stride + 1, self.stride),
            paddle.arange(1, w * self.stride + 1, self.stride))
        p_0_x = paddle.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = paddle.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = paddle.concat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = paddle.concat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             axis=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

