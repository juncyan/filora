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

import warnings
from collections import OrderedDict
import numpy as np
from paddle.static import Program, Variable
import paddle
from paddle import nn
from paddle.jit.dy2static.program_translator import unwrap_decorators

__all__ = []

def GradCAM(model, input_data, class_idx):  
    model.eval()  
    with paddle.no_grad():  
        logits = model(input_data)  
    grads = paddle.grad(logits[:, class_idx], model.parameters(), create_graph=True)  
    weights = []  
    for g in grads:  
        g = g.numpy()  
        weights += list(g[0])  
    weights = np.mean(weights, axis=(0, 1))  
    cam = np.zeros(input_data.shape[1:], dtype=np.float32)  
    for i, w in enumerate(weights):  
        cam += w * input_data[0, i]  
    cam = cam - np.max(cam)  
    cam = cam / np.max(cam)  
    cam = np.uint8(255 * cam)  
    return cam

def flops(net, input_sizes, num_inputs=1, custom_ops=None, print_detail=False):
    """Print a table about the FLOPs of network.

    Args:
        net (paddle.nn.Layer||paddle.static.Program): The network which could be a instance of paddle.nn.Layer in
                    dygraph or paddle.static.Program in static graph.
        input_sizes(list): size of input tensor.
        num_inputs(int): numbers of inputs, 1 or 2
        custom_ops (A dict of function, optional): A dictionary which key is the class of specific operation such as
                    paddle.nn.Conv2D and the value is the function used to count the FLOPs of this operation. This
                    argument only work when argument ``net`` is an instance of paddle.nn.Layer. The details could be found
                    in following example code. Default is None.
        print_detail (bool, optional): Whether to print the detail information, like FLOPs per layer, about the net FLOPs.
                    Default is False.

    Returns:
        Int: A number about the FLOPs of total network.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            class LeNet(nn.Layer):
                def __init__(self, num_classes=10):
                    super().__init__()
                    self.num_classes = num_classes
                    self.features = nn.Sequential(
                        nn.Conv2D(
                            1, 6, 3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2D(2, 2),
                        nn.Conv2D(
                            6, 16, 5, stride=1, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2D(2, 2))

                    if num_classes > 0:
                        self.fc = nn.Sequential(
                            nn.Linear(400, 120),
                            nn.Linear(120, 84),
                            nn.Linear(
                                84, 10))

                def forward(self, inputs):
                    x = self.features(inputs)

                    if self.num_classes > 0:
                        x = paddle.flatten(x, 1)
                        x = self.fc(x)
                    return x

            lenet = LeNet()
            # m is the instance of nn.Layer, x is the intput of layer, y is the output of layer.
            def count_leaky_relu(m, x, y):
                x = x[0]
                nelements = x.numel()
                m.total_ops += int(nelements)

            FLOPs = paddle.flops(lenet, [1, 1, 28, 28], custom_ops= {nn.LeakyReLU: count_leaky_relu},
                                print_detail=True)
            print(FLOPs)

            #+--------------+-----------------+-----------------+--------+--------+
            #|  Layer Name  |   Input Shape   |   Output Shape  | Params | Flops  |
            #+--------------+-----------------+-----------------+--------+--------+
            #|   conv2d_2   |  [1, 1, 28, 28] |  [1, 6, 28, 28] |   60   | 47040  |
            #|   re_lu_2    |  [1, 6, 28, 28] |  [1, 6, 28, 28] |   0    |   0    |
            #| max_pool2d_2 |  [1, 6, 28, 28] |  [1, 6, 14, 14] |   0    |   0    |
            #|   conv2d_3   |  [1, 6, 14, 14] | [1, 16, 10, 10] |  2416  | 241600 |
            #|   re_lu_3    | [1, 16, 10, 10] | [1, 16, 10, 10] |   0    |   0    |
            #| max_pool2d_3 | [1, 16, 10, 10] |  [1, 16, 5, 5]  |   0    |   0    |
            #|   linear_0   |     [1, 400]    |     [1, 120]    | 48120  | 48000  |
            #|   linear_1   |     [1, 120]    |     [1, 84]     | 10164  | 10080  |
            #|   linear_2   |     [1, 84]     |     [1, 10]     |  850   |  840   |
            #+--------------+-----------------+-----------------+--------+--------+
            #Total Flops: 347560     Total Params: 61610
    """
    if isinstance(net, nn.Layer):
        # If net is a dy2stat model, net.forward is StaticFunction instance,
        # we set net.forward to original forward function.
        # _, net.forward = unwrap_decorators(net.forward)

        img = paddle.rand(input_sizes)
        inputs = num_inputs * [img]
        return dynamic_flops(
            net, inputs=inputs, custom_ops=custom_ops, print_detail=print_detail
        )
    elif isinstance(net, paddle.static.Program):
        return static_flops(net, print_detail=print_detail)
    else:
        warnings.warn(
            "Your model must be an instance of paddle.nn.Layer or paddle.static.Program."
        )
        return -1


def count_convNd(m, x, y):
    x = x[0]
    kernel_ops = np.product(m.weight.shape[2:])
    bias_ops = 1 if m.bias is not None else 0
    total_ops = int(y.numel()) * (
        x.shape[1] / m._groups * kernel_ops + bias_ops
    )
    m.total_ops += abs(int(total_ops))


def count_leaky_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += int(nelements)


def count_bn(m, x, y):
    x = x[0]
    nelements = x.numel()
    if not m.training:
        total_ops = 2 * nelements
    m.total_ops += abs(int(total_ops))


def count_linear(m, x, y):
    total_mul = m.weight.shape[0]
    num_elements = y.numel()
    total_ops = total_mul * num_elements
    m.total_ops += abs(int(total_ops))


def count_avgpool(m, x, y):
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += int(total_ops)


def count_adap_avgpool(m, x, y):
    kernel = np.array(x[0].shape[2:]) // np.array(y.shape[2:])
    total_add = np.product(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops += abs(int(total_ops))


def count_zero_ops(m, x, y):
    m.total_ops += int(0)


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += p.numel()
    m.total_params[0] = abs(int(total_params))


def count_io_info(m, x, y):
    # print("Input shape: ", x[0])
    if type(x[0]) is tuple:
        inshape = x[0]
    else:
        inshape = x[0].shape
    m.register_buffer('input_shape', paddle.to_tensor(inshape))
    if isinstance(y, (list, tuple)):
        m.register_buffer('output_shape', paddle.to_tensor(y[0].shape))
    else:
        m.register_buffer('output_shape', paddle.to_tensor(y.shape))


register_hooks = {
    nn.Conv1D: count_convNd,
    nn.Conv2D: count_convNd,
    nn.Conv3D: count_convNd,
    nn.Conv1DTranspose: count_convNd,
    nn.Conv2DTranspose: count_convNd,
    nn.Conv3DTranspose: count_convNd,
    nn.layer.norm.BatchNorm2D: count_bn,
    nn.BatchNorm: count_bn,
    nn.ReLU: count_zero_ops,
    nn.ReLU6: count_zero_ops,
    nn.LeakyReLU: count_leaky_relu,
    nn.Linear: count_linear,
    nn.Dropout: count_zero_ops,
    nn.AvgPool1D: count_avgpool,
    nn.AvgPool2D: count_avgpool,
    nn.AvgPool3D: count_avgpool,
    nn.AdaptiveAvgPool1D: count_adap_avgpool,
    nn.AdaptiveAvgPool2D: count_adap_avgpool,
    nn.AdaptiveAvgPool3D: count_adap_avgpool,
}


def dynamic_flops(model, inputs,custom_ops=None, print_detail=False):
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('total_params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)

        flops_fn = None
        if m_type in custom_ops:
            flops_fn = custom_ops[m_type]
            if m_type not in types_collection:
                print(f"Customize Function has been applied to {m_type}")
        elif m_type in register_hooks:
            flops_fn = register_hooks[m_type]
            if m_type not in types_collection:
                print(f"{m_type}'s flops has been counted")
        else:
            if m_type not in types_collection:
                print(
                    "Cannot find suitable count function for {}. Treat it as zero FLOPs.".format(
                        m_type
                    )
                )

        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)
        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with paddle.framework.no_grad():
        if len(inputs) == 1:
            model(inputs[0])
        elif len(inputs) == 2:
            model(inputs[0], inputs[1])
        else:
            warnings.warn(
            "Your model must be an instance of paddle.nn.Layer or paddle.static.Program."
        )
            

    total_ops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if {
            'total_ops',
            'total_params',
            'input_shape',
            'output_shape',
        }.issubset(set(m._buffers.keys())):
            total_ops += m.total_ops
            total_params += m.total_params

    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()

    table = Table(
        ["Layer Name", "Input Shape", "Output Shape", "Params", "Flops"]
    )

    for n, m in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue
        if {
            'total_ops',
            'total_params',
            'input_shape',
            'output_shape',
        }.issubset(set(m._buffers.keys())):
            table.add_row(
                [
                    m.full_name(),
                    list(m.input_shape.numpy()),
                    list(m.output_shape.numpy()),
                    int(m.total_params),
                    int(m.total_ops),
                ]
            )
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')
    if print_detail:
        table.print_table()
    print(
        'Total Flops: {}     Total Params: {}'.format(
            int(total_ops), int(total_params)
        )
    )
    return {"total_ops":int(total_ops), "total_params": int(total_params)}

class VarWrapper:
    def __init__(self, var, graph):
        assert isinstance(var, Variable)
        assert isinstance(graph, GraphWrapper)
        self._var = var
        self._graph = graph

    def name(self):
        """
        Get the name of the variable.
        """
        return self._var.name

    def shape(self):
        """
        Get the shape of the varibale.
        """
        return self._var.shape


class OpWrapper:
    def __init__(self, op, graph):
        assert isinstance(graph, GraphWrapper)
        self._op = op
        self._graph = graph

    def type(self):
        """
        Get the type of this operator.
        """
        return self._op.type

    def inputs(self, name):
        """
        Get all the varibales by the input name.
        """
        if name in self._op.input_names:
            return [
                self._graph.var(var_name) for var_name in self._op.input(name)
            ]
        else:
            return []

    def outputs(self, name):
        """
        Get all the varibales by the output name.
        """
        return [self._graph.var(var_name) for var_name in self._op.output(name)]


class GraphWrapper:
    """
    It is a wrapper of paddle.fluid.framework.IrGraph with some special functions
    for paddle slim framework.

    Args:
        program(framework.Program): A program with
        in_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
        out_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
    """

    def __init__(self, program=None, in_nodes=[], out_nodes=[]):
        """ """
        super().__init__()
        self.program = Program() if program is None else program
        self.persistables = {}
        self.teacher_persistables = {}
        for var in self.program.list_vars():
            if var.persistable:
                self.persistables[var.name] = var
        self.compiled_graph = None
        in_nodes = [] if in_nodes is None else in_nodes
        out_nodes = [] if out_nodes is None else out_nodes
        self.in_nodes = OrderedDict(in_nodes)
        self.out_nodes = OrderedDict(out_nodes)
        self._attrs = OrderedDict()

    def ops(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        ops = []
        for block in self.program.blocks:
            for op in block.ops:
                ops.append(OpWrapper(op, self))
        return ops

    def var(self, name):
        """
        Get the variable by variable name.
        """
        for block in self.program.blocks:
            if block.has_var(name):
                return VarWrapper(block.var(name), self)
        return None


def count_convNd(op):
    filter_shape = op.inputs("Filter")[0].shape()
    filter_ops = np.product(filter_shape[1:])
    bias_ops = 1 if len(op.inputs("Bias")) > 0 else 0
    output_numel = np.product(op.outputs("Output")[0].shape()[1:])
    total_ops = output_numel * (filter_ops + bias_ops)
    total_ops = abs(total_ops)
    return total_ops


def count_leaky_relu(op):
    total_ops = np.product(op.outputs("Output")[0].shape()[1:])
    return total_ops


def count_bn(op):
    output_numel = np.product(op.outputs("Y")[0].shape()[1:])
    total_ops = 2 * output_numel
    total_ops = abs(total_ops)
    return total_ops


def count_linear(op):
    total_mul = op.inputs("Y")[0].shape()[0]
    numel = np.product(op.outputs("Out")[0].shape()[1:])
    total_ops = total_mul * numel
    total_ops = abs(total_ops)
    return total_ops


def count_pool2d(op):
    input_shape = op.inputs("X")[0].shape()
    output_shape = op.outputs('Out')[0].shape()
    kernel = np.array(input_shape[2:]) // np.array(output_shape[2:])
    total_add = np.product(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = np.product(output_shape[1:])
    total_ops = kernel_ops * num_elements
    total_ops = abs(total_ops)
    return total_ops


def count_element_op(op):
    input_shape = op.inputs("X")[0].shape()
    total_ops = np.product(input_shape[1:])
    total_ops = abs(total_ops)
    return total_ops


def _graph_flops(graph, detail=False):
    assert isinstance(graph, GraphWrapper)
    flops = 0
    op_flops = 0
    table = Table(["OP Type", 'Param name', "Flops"])
    for op in graph.ops():
        param_name = ''
        if op.type() in ['conv2d', 'depthwise_conv2d']:
            op_flops = count_convNd(op)
            flops += op_flops
            param_name = op.inputs("Filter")[0].name()
        elif op.type() == 'pool2d':
            op_flops = count_pool2d(op)
            flops += op_flops

        elif op.type() in ['mul', 'matmul']:
            op_flops = count_linear(op)
            flops += op_flops
            param_name = op.inputs("Y")[0].name()
        elif op.type() == 'batch_norm':
            op_flops = count_bn(op)
            flops += op_flops
        elif op.type().startswith('element'):
            op_flops = count_element_op(op)
            flops += op_flops
        if op_flops != 0:
            table.add_row([op.type(), param_name, op_flops])
        op_flops = 0
    if detail:
        table.print_table()
    return flops


def static_flops(program, print_detail=False):
    graph = GraphWrapper(program)
    return _graph_flops(graph, detail=print_detail)


class Table:
    def __init__(self, table_heads):
        self.table_heads = table_heads
        self.table_len = []
        self.data = []
        self.col_num = len(table_heads)
        for head in table_heads:
            self.table_len.append(len(head))

    def add_row(self, row_str):
        if not isinstance(row_str, list):
            print('The row_str should be a list')
        if len(row_str) != self.col_num:
            print(
                'The length of row data should be equal the length of table heads, but the data: {} is not equal table heads {}'.format(
                    len(row_str), self.col_num
                )
            )
        for i in range(self.col_num):
            if len(str(row_str[i])) > self.table_len[i]:
                self.table_len[i] = len(str(row_str[i]))
        self.data.append(row_str)

    def print_row(self, row):
        string = ''
        for i in range(self.col_num):
            string += '|' + str(row[i]).center(self.table_len[i] + 2)
        string += '|'
        print(string)

    def print_shelf(self):
        string = ''
        for length in self.table_len:
            string += '+'
            string += '-' * (length + 2)
        string += '+'
        print(string)

    def print_table(self):
        self.print_shelf()
        self.print_row(self.table_heads)
        self.print_shelf()
        for data in self.data:
            self.print_row(data)
        self.print_shelf()
   

    
