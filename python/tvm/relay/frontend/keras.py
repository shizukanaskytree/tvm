# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, import-self, import-outside-toplevel
"""Keras frontend."""
import sys
import numpy as np
import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
# python/tvm/relay/op/__init__.py ↑
from ... import nd as _nd
from .common import ExprTable, new_var

__all__ = ['from_keras']
# 1.
# 当别人 from python/tvm/relay/frontend/keras.py import * 时
# 只能从这个文件中 export 出 from_keras 这个函数. 其他的对外不可见.
#
# def from_keras(model, shape=None, layout='NCHW'):
#   """Convert keras model to relay Function.


def _check_data_format(keras_layer):
    if hasattr(keras_layer, ('data_format')):
        if keras_layer.data_format != 'channels_last':
            raise ValueError("Keras frontend currently supports data_format = channels_last only.")


def _get_pad_pair(input1d, kernel1d, stride1d):
    out1d = (input1d + stride1d - 1) // stride1d
    pad = np.maximum((out1d - 1) * stride1d + kernel1d - input1d, 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return [pad_before, pad_after]


def _get_elu(inexpr, alpha):
    """A helper method for elu."""
    return _op.negative(alpha) * _op.nn.relu(_expr.const(1., dtype='float32') - \
        _op.exp(inexpr)) + _op.nn.relu(inexpr)


def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]


def _convert_recurrent_activation(inexpr, keras_layer):
    act_type = keras_layer.recurrent_activation.__name__
    return _convert_activation(inexpr, act_type, None)


def _convert_activation(inexpr, keras_layer, _):
    # 1.
    # Activation
    # conv1_relu (Activation)  (None, 112, 112, 64) 0  conv1_bn[0][0]
    # example: https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5#file-expression_table_names-log-L44

    # 2.
    # inexpr value:
    # https://gist.github.com/shizukanaskytree/5c3579cf2f2bbfe49249837260c59d8c

    # 2.1
    # 由于上面没有 nn.relu, 所以我又 double-check 了一下:
    # https://gist.github.com/shizukanaskytree/4a7f59319bb831006a0feb4b25b039f2#file-etable_name_out-log-L506
    # 解释: 可能是传入前的所以 nn.relu 还没构造出来.

    # 3.
    # keras_layer value:
    # <tensorflow.python.keras.layers.core.Activation object at 0x7fbe116f5898>

    # 4.
    # _ は?
    # _ : <tvm.relay.frontend.common.ExprTable object at 0x7fbdd43152b0>

    if isinstance(keras_layer, str):
        # 未进入
        act_type = keras_layer
    else:
        # 进入
        if sys.version_info.major < 3:
            # 未进入
            # 1.
            # sys.version_info.major == 3
            act_type = keras_layer.activation.func_name
        else:
            # 进入
            act_type = keras_layer.activation.__name__
            # 1.
            # act_type value:
            # 'relu'

    if act_type == 'linear':
        if isinstance(keras_layer, str):
            return inexpr
        alpha = keras_layer.alpha if hasattr(keras_layer, 'alpha') else 1.
        beta = keras_layer.beta if hasattr(keras_layer, 'beta') else 0.
        alpha = _expr.const(alpha, dtype='float32')
        beta = _expr.const(beta, dtype='float32')
        return _op.add(_op.multiply(inexpr, alpha), beta)
    if act_type == 'softmax':
        return _op.nn.softmax(inexpr, axis=1)
    if act_type == 'sigmoid':
        return _op.sigmoid(inexpr)
    if act_type == 'tanh':
        return _op.tanh(inexpr)
    if act_type == 'relu':
        # 1.
        # relu 实例:
        # https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5#file-expression_table_names-log-L44

        return _op.nn.relu(inexpr)
        # 1.
        # inexpr value:
        # https://gist.github.com/shizukanaskytree/5c3579cf2f2bbfe49249837260c59d8c

        # 1.1
        # 由于上面没有 nn.relu, 所以我又 double-check 了一下:
        # https://gist.github.com/shizukanaskytree/4a7f59319bb831006a0feb4b25b039f2#file-etable_name_out-log-L506
        # 解释: 可能是传入前的所以 nn.relu 还没构造出来.

        # 2.
        # # python/tvm/relay/op/nn/nn.py ↓
        # def relu(data):
        # - data: tvm.relay.Expr, The input data.

        # 3.
        # 🛑 ⽌ since `return`

        # 4.
        # return to python/tvm/relay/frontend/keras.py:
        # def keras_op_to_relay(inexpr, keras_layer, outname, etab):
        #   outs = _convert_map[op_name](inexpr, keras_layer, etab)
        #   ↳ 没错, 就是这里, 这里是上面这个完成后的首末站!
        #
        # 4.1
        # 那时的结果是:
        # https://gist.github.com/shizukanaskytree/735e0c30df0dbc58e55d7deeef309041
        # - nn.relu(%4)

    if act_type == 'softplus':
        return _op.log(_op.add(_op.exp(inexpr), _expr.const(1., dtype='float32')))
    if act_type == 'elu':
        alpha = keras_layer.alpha if hasattr(keras_layer, 'alpha') else 1.
        alpha = _expr.const(alpha, dtype='float32')
        return _get_elu(inexpr, alpha)
    if act_type == 'selu':
        # Alpha, Gamma values obtained from https://arxiv.org/abs/1706.02515
        alpha = keras_layer.alpha if hasattr(keras_layer, 'alpha') \
            else 1.6732632423543772848170429916717
        gamma = keras_layer.gamma if hasattr(keras_layer, 'gamma') \
            else 1.0507009873554804934193349852946
        alpha = _expr.const(alpha, dtype='float32')
        gamma = _expr.const(gamma, dtype='float32')
        return gamma * _get_elu(inexpr, alpha)
    if act_type == 'relu6':
        return _op.clip(inexpr, a_min=0., a_max=6.)
    if act_type == 'softsign':
        return inexpr / (_expr.const(1., dtype='float32') + _op.abs(inexpr))
        # 1.
        # _op.abs path:
        # python/tvm/relay/op/tensor.py

        # 2.
        # QQQ: inexpr 为什么可以 除以一个 _expr.const ???
        # AAA:

    if act_type == 'hard_sigmoid':
        x = (_expr.const(0.2, dtype='float32') * inexpr) + _expr.const(0.5, dtype='float32')
        # 1.
        # QQQ: _expr.const(0.2, dtype='float32') * inexpr 为什么可以乘起来???
        # AAA:

        return _op.clip(x, a_min=0., a_max=1.)

    raise tvm.error.OpNotImplemented(
        'Operator {} is not supported in frontend Keras.'.format(act_type))


def _convert_advanced_activation(inexpr, keras_layer, etab):
    act_type = type(keras_layer).__name__

    if act_type == 'Softmax':
        axis = keras_layer.axis
        dims = len(keras_layer.input_shape)
        if isinstance(axis, list):
            raise tvm.error.OpAttributeUnImplemented(
                'Softmax with axes {} is not supported.'.format(axis))
        if axis == -1:
            axis = 1
        else:
            axis = axis + 1 if axis < dims - 1 else 1
        return _op.nn.softmax(inexpr, axis=axis)
        # 1.
        # _op.nn.softmax path:
        # relay.op.nn.nn.softmax

    if act_type == 'ReLU':
        threshold = _expr.const(keras_layer.threshold, dtype='float32')
        # 1.
        # _expr.const path:
        # relay.expr.const

        if keras_layer.max_value and float(keras_layer.threshold) == 0:
            # f(x) = max_value, for x >= max_value
            # f(x) = x,         for threshold <= x < max_value
            return _op.clip(inexpr, a_min=0., a_max=float(keras_layer.max_value))
            # 1.
            # _op.clip path:
            # relay.op.tensor.clip

        if keras_layer.max_value and _op.greater(threshold, inexpr).astype('float32'):
            # f(x) = negative_slope * (inexpr - threshold)
            negative_slope = _expr.const(keras_layer.negative_slope, dtype='float32')
            return _op.multiply(negative_slope, _op.subtract(inexpr, threshold))
            # 1.
            # _op.multiply path:
            # relay.op.tensor.multiply

        return _op.nn.relu(inexpr)
    if act_type == 'LeakyReLU':
        return _op.nn.leaky_relu(inexpr, alpha=float(keras_layer.alpha))
    if act_type == 'ELU':
        alpha = keras_layer.alpha if hasattr(keras_layer, 'alpha') else 1.
        alpha = _expr.const(alpha, dtype='float32')
        return _get_elu(inexpr, alpha)
    if act_type == 'PReLU':
        assert hasattr(keras_layer, 'alpha'), "alpha required for PReLU."
        _check_data_format(keras_layer)
        size = len(keras_layer.alpha.shape)
        alpha = etab.new_const(keras_layer.get_weights()[0] \
                               .transpose(np.roll(range(size), 1)))
        return _op.negative(alpha) * _op.nn.relu(_op.negative(inexpr)) + _op.nn.relu(inexpr)
        # 1.
        # _op.negative path:
        # relay.op.tensor.negative

    if act_type == 'ThresholdedReLU':
        theta = keras_layer.theta if hasattr(keras_layer, 'theta') else 1.
        return _op.multiply(inexpr, _op.greater(inexpr, \
            _expr.const(theta, dtype='float32')).astype('float32'))

    raise tvm.error.OpNotImplemented(
        'Operator {} is not supported in frontend Keras.'.format(act_type))


def _convert_merge(inexpr, keras_layer, _):
    merge_type = type(keras_layer).__name__
    ret = inexpr[0]
    if merge_type == 'Dot':
        axes = keras_layer.axes
        if isinstance(keras_layer.axes, int):
            axes = [keras_layer.axes, keras_layer.axes]
        if isinstance(axes, list):
            if len(axes) != 2:
                raise tvm.error.OpAttributeUnImplemented(
                    'Dot with axes {} is not supported.'.format(keras_layer.axes))
            for i, axis in enumerate(axes):
                if axis not in [1, 2]:
                    raise tvm.error.OpAttributeUnImplemented(
                        'Dot with axes {} is not supported.'.format(keras_layer.axes))
                if axes[i] == 2:
                    inexpr[i] = _op.transpose(inexpr[i], axes=[0, 2, 1])
        else:
            raise tvm.error.OpAttributeUnImplemented(
                'Dot with axes {} is not supported.'.format(keras_layer.axes))
        ret_dot = _op.nn.batch_matmul(inexpr[0], inexpr[1])
        ret = _op.transpose(ret_dot, axes=[0, 2, 1])
    elif merge_type == 'Subtract':
        assert len(inexpr) == 2, "Subtract merge takes 2 inputs."
        ret = _op.subtract(ret, inexpr[1])
    elif merge_type in ['Add', 'Multiply', 'Minimum', 'Maximum']:
        op_map = {'Add': _op.add,
                  'Multiply': _op.multiply,
                  'Minimum': _op.minimum,
                  'Maximum': _op.maximum}
        for i in range(1, len(inexpr)):
            ret = op_map[merge_type](ret, inexpr[i])
    elif merge_type == 'Average':
        for i in range(1, len(inexpr)):
            ret = _op.add(ret, inexpr[i])
        ret = ret / _expr.const(len(inexpr), dtype='float32')
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported in frontend Keras.'.format(merge_type))
    return ret


def _convert_permute(inexpr, keras_layer, _):
    return _op.transpose(inexpr, axes=(0,) + keras_layer.dims)


def _convert_embedding(inexpr, keras_layer, etab):
    indices = inexpr
    weightList = keras_layer.get_weights()
    weight = etab.new_const(weightList[0])
    out = _op.take(weight, indices.astype('int32'), axis=0)

    return out

def _convert_dense(inexpr, keras_layer, etab):
    weightList = keras_layer.get_weights()
    # 1.
    # SO: Keras: Interpreting the output of get_weights() of dense layer in keras.
    # https://stackoverflow.com/questions/46817085/keras-interpreting-the-output-of-get-weights

    weight = etab.new_const(weightList[0].transpose([1, 0]))
    params = {'weight': weight, 'units': weightList[0].shape[1]}
    input_shape = keras_layer.input_shape
    input_dim = len(input_shape)
    # In case of RNN dense, input shape will be (1, 1, n)
    if input_dim > 2:
        input_shape = tuple(dim if dim else 1 for dim in _as_list(input_shape)[0])
        if input_dim != 3 or input_shape[0] != 1 or input_shape[1] != 1:
            raise tvm.error.OpAttributeInvalid(
                'Input shape {} is not valid for operator Dense.'.format(input_shape))
        inexpr = _op.squeeze(inexpr, axis=0)
    out = _op.nn.dense(data=inexpr, **params)
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        out = _op.nn.bias_add(out, bias)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, etab)
    if input_dim > 2:
        out = _op.expand_dims(out, axis=0)
    return out


def _convert_convolution(inexpr, keras_layer, etab):
    # 1.
    # called from :
    # # python/tvm/relay/frontend/keras.py ↓
    # outs = _convert_map[op_name](inexpr, keras_layer, etab)

    # 1.1
    # op_name = "Conv2D"

    # 2.
    # arguments:
    # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198#file-out_head_1500-log-L401-L410

    # 2.1
    # inexpr value:
    # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198#file-out_head_1500-log-L405-L407

    _check_data_format(keras_layer)
    is_deconv = type(keras_layer).__name__ == 'Conv2DTranspose'
    # 1.
    # type(keras_layer) value:
    # <class 'tensorflow.python.keras.layers.convolutional.Conv2D'>
    #
    # 1.1
    # type(keras_layer).__name__ value:
    # 'Conv2D'

    # 2.
    # is_deconv value:
    # False

    is_depthconv = type(keras_layer).__name__ == 'DepthwiseConv2D'
    # 1.
    # is_depthconv value:
    # False

    weightList = keras_layer.get_weights()
    # 1.
    # weightList value:
    # https://gist.github.com/shizukanaskytree/07386d387ee60f0e4494519367545d5c

    # 1.1
    # weightList shape:
    # 1. (7,7,3,64) -- 64 个 filters
    # 2. (64,) -- 64 个 bias, 施加(+ / add)到 每层 output 的 bias 都是同样的值
    # https://keep.google.com/u/1/#NOTE/1VbnxLbsKE1MJny0YLKTQJa4oy2YwVInZQLvkIXZCE-7wM0ELfouFv-_azNOkbA
    # comment: 具体为什么是这个 HWCN 格式我没有查到文档说明.

    # 1.1.1
    # Q: Keras get_weights 输出形状 为什么是 HWCN 格式?
    # https://keras.io/zh/layers/about-keras-layers/
    # A:
    # layer.get_weights(): 以含有Numpy矩阵的列表形式返回层的权重。
    # layer.set_weights(weights): 从含有Numpy矩阵的列表中设置层的权重（与get_weights的输出形状相同）。

    # 1.1.2
    # Keras ResNet50 code:
    # 1st Conv2D:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py#L226-L230
    #
    # all code:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

    # 1.1.1
    # Keras layers API
    # - https://keras.io/api/layers/
    # 其中的例子, 你看!
    #  [<tf.Variable 'dense/kernel:0' shape=(20, 32) dtype=float32>,
    #  <tf.Variable 'dense/bias:0' shape=(32,) dtype=float32>]

    # 1.1.1
    # weights property in The base Layer class:
    # https://keras.io/api/layers/base_layer/#weights-property

    # 1.1.2
    # get_weights: tf keras API: tf.keras.layers.Layer | TensorFlow Core v2.2.0
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#get_weights

    # 1.1.3
    # SO: How do I get the weights of a layer in Keras?
    # https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras

    # 1.1.4
    # ResNet Convolution 推导和图示:
    # Figure 4. Conv1 — Convolution
    # Figure 6. Layer 1, block 1, operation 1
    # in the post:
    # https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
    # https://keras.io/guides/functional_api/ <== 自己打印啦

    # 1.1.5
    # ResNet50 keras application github code:
    # 目的是找到为什么 weights 是按照 (7,7,3,64) 这样排布的或者输出的?
    # ResNet50 each layer's shape

    # 1.1.6
    # 相关可视化的链接
    # https://keras.io/guides/functional_api/
    # https://cloud.tencent.com/developer/article/1065135
    # https://zhuanlan.zhihu.com/p/24833574

    # 1.2
    # weightList type:
    # list, len=2, within it is np.array

    # 2.
    # Convolution layer weights in keras は？
    # https://stackoverflow.com/questions/43305891/how-to-correctly-get-layer-weights-from-conv2d-in-keras
    # weight 就是 那些 filters 里面的值啊, 就是上面 SO 链接里面的那些黑色, 灰色的那些值啊.
    # 上面那个帖子里面只显示了 25 个 filter 的 case .

    # 2.1
    # Keras Conv2D API:
    # https://keras.io/api/layers/convolution_layers/convolution2d/
    # Conv2D(filters, ...) 里面的 filters 是指
    # The convolution layer comprises of a set of independent filters (6 in the example shown).
    # shown in fig 5 in
    # https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8

    # 3.
    # Convolution explain in CNN
    # https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8

    weight = weightList[0]
    # 1.
    # 取出 weights 那个
    # (7,7,3,64) -- 64 个 filters

    if etab.data_layout == 'NHWC':
        # 1.
        # etab.data_layout value:
        # 'NCHW'

        # 未进入!
        if is_depthconv:
            kernel_layout = 'HWOI'
        else:
            kernel_layout = 'HWIO'
    else:
        # 进入这个分支!
        kernel_layout = 'OIHW'
        # 1.
        # I: input channels
        # 教程: https://oneapi-src.github.io/oneDNN/understanding_memory_formats.html

    if is_deconv:
        kernel_h, kernel_w, n_filters, in_channels = weight.shape
        if kernel_layout == 'OIHW':
            weight = weight.transpose([3, 2, 0, 1])
    elif is_depthconv:
        kernel_h, kernel_w, in_channels, depth_mult = weight.shape
        if kernel_layout == 'OIHW':
            weight = weight.transpose([2, 3, 0, 1])
    elif etab.data_layout == 'NCHW':
        # 进入!

        # 1.
        # QQQ: 为什么 'NCHW' 对应如下时却是 (h w c n) ?
        # AAA: 可能说目标是 'NCHW' 但是输入其实是 'HWCN'

        kernel_h, kernel_w, in_channels, n_filters = weight.shape
        # 1.
        # weight.shape value:
        # (7,7,3,64)
        # 所以如下才进行了挪位处理.

        weight = weight.transpose([3, 2, 0, 1])
    else:
        kernel_h, kernel_w, in_channels, n_filters = weight.shape

    if isinstance(keras_layer.dilation_rate, (list, tuple)):
        # 1.
        # 进入
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
        # 1.
        # keras_layer.dilation_rate value:
        # (1,1)
        # - dilation rate k=1 is normal convolution

        # 2.
        # Explain Dilated Convolution:
        # 扩大的；膨胀的；加宽的
        # https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25

        # 3.
        # Explain dilation_rate concept in conv2d:
        # https://erogol.com/dilated-convolution/
        # https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
        #
        # dilation rate k
        # k=1 means normal convolution. Fig (a)
        # k=2 means skipping 1 pixels. Fig (b)
        # k=4 means skipping 3 pixels. Fig (c)
        # https://erogol.com/dilated-convolution/#:~:text=Dilated%20Convolution,4%20means%20skipping%203%20pixels.

        # 3.1
        # Experiment
        # https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25
        # ↑ 上面这个帖子先用 numpy 硬撸一遍得到结果作为预期, 再用 tf code 实现一遍对比结果.
        # 代码: https://gist.github.com/shizukanaskytree/1500d791d5420b2597373e0bd24cf47f

    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]

    dilated_kernel_h = (kernel_h - 1) * dilation[0] + 1
    # 1.
    # value:
    # 7
    dilated_kernel_w = (kernel_w - 1) * dilation[1] + 1
    # 1.
    # value:
    # 7
    stride_h, stride_w = keras_layer.strides
    # 1.
    # value:
    # 2, 2
    # double-check code:
    # https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py#L227

    params = {'weight': etab.new_const(weight),
              'kernel_size': [kernel_h, kernel_w],
              'strides': [stride_h, stride_w],
              'dilation': dilation,
              'padding': [0, 0],
              'data_layout': etab.data_layout,
              'kernel_layout': kernel_layout}
    # 1.
    # params value:
    # - weight:
    #   {ndarray:(64,3,7,7)}
    #   Value: https://gist.github.com/shizukanaskytree/07386d387ee60f0e4494519367545d5c
    # - [kernel_h, kernel_w]=[7,7]
    # - [stride_h, stride_w]=[2,2]
    # - dilation=(1,1)
    # - etab.data_layout='NCHW'
    # - kernel_layout='OIHW'
    #
    # params value:
    # {'weight': Var(_param_1, ty=TensorType([64, 3, 7, 7], float32)), 'kernel_size': [7, 7], 'strides': [2, 2], 'dilation': [1, 1], 'padding': [0, 0], 'data_layout': 'NCHW', 'kernel_layout': 'OIHW'}

    if is_depthconv:
        params['channels'] = in_channels * depth_mult
        params['groups'] = in_channels
    else:
        params['channels'] = n_filters
        # 1.
        # n_filters: numbers of filters = 64

    if keras_layer.padding == 'valid':
        # 进入
        pass
    # we insert a separate pad operator
    elif keras_layer.padding == 'same':
        in_h = keras_layer.input_shape[1]
        in_w = keras_layer.input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        params['padding'] = (pad_t, pad_l, pad_b, pad_r)
    else:
        msg = 'Padding with {} is not supported for operator Convolution ' \
              'in frontend Keras.'
        raise tvm.error.OpAttributeUnImplemented(msg.format(keras_layer.padding))

    if is_deconv:
        out = _op.nn.conv2d_transpose(data=inexpr, **params)
    else:
        out = _op.nn.conv2d(data=inexpr, **params)
        # 1.
        # ✅ 关键的最终步: 外对内传入 _op.nn.conv2d

        # 2.1
        # inexpr value:
        # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198#file-out_head_1500-log-L405-L407

    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        if etab.data_layout == 'NCHW':
            out = _op.nn.bias_add(out, bias)
        else:
            out = _op.nn.bias_add(out, bias, axis=-1)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, etab)
    return out

def _convert_convolution3d(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    weightList = keras_layer.get_weights()
    weight = weightList[0]

    if etab.data_layout == 'NDHWC':
        kernel_layout = 'DHWIO'
    else:
        kernel_layout = 'OIDHW'
        msg = 'Kernel layout with {} is not supported for operator Convolution3D ' \
              'in frontend Keras.'
        raise tvm.error.OpAttributeUnImplemented(msg.format(etab.data_layout))

    dilation_rate = keras_layer.dilation_rate
    if isinstance(dilation_rate, (list, tuple)):
        dilation = [dilation_rate[0], dilation_rate[1], dilation_rate[2]]
    else:
        dilation = [dilation_rate, dilation_rate, dilation_rate]

    kernel_d1 = weight.shape[0]
    kernel_d2 = weight.shape[1]
    kernel_d3 = weight.shape[2]
    # in_channels = weight.shape[3]
    n_filters = weight.shape[4]

    dilated_kernel_d1 = (kernel_d1 - 1) * dilation[0] + 1
    dilated_kernel_d2 = (kernel_d2 - 1) * dilation[1] + 1
    dilated_kernel_d3 = (kernel_d3 - 1) * dilation[2] + 1
    stride_d1, stride_d2, stride_d3 = keras_layer.strides
    params = {'weight': etab.new_const(weight),
              'kernel_size': [kernel_d1, kernel_d2, kernel_d3],
              'strides': [stride_d1, stride_d2, stride_d3],
              'dilation': dilation,
              'padding': [0, 0, 0],
              'data_layout': etab.data_layout,
              'kernel_layout': kernel_layout}
    params['channels'] = n_filters

    if keras_layer.padding == 'valid':
        pass
    # calculate the padding values
    elif keras_layer.padding == 'same':
        in_d1 = keras_layer.input_shape[1]
        in_d2 = keras_layer.input_shape[2]
        in_d3 = keras_layer.input_shape[3]
        pad_d1 = _get_pad_pair(in_d1, dilated_kernel_d1, stride_d1)
        pad_d2 = _get_pad_pair(in_d2, dilated_kernel_d2, stride_d2)
        pad_d3 = _get_pad_pair(in_d3, dilated_kernel_d3, stride_d3)
        params['padding'] = [pad_d1[0], pad_d2[0], pad_d3[0], pad_d1[1], pad_d2[1], pad_d3[1]]
    else:
        msg = 'Padding with {} is not supported for operator Convolution3D ' \
              'in frontend Keras.'
        raise tvm.error.OpAttributeUnImplemented(msg.format(keras_layer.padding))
    out = _op.nn.conv3d(data=inexpr, **params)

    channel_axis = -1 if etab.data_layout == "NDHWC" else 1
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        out = _op.nn.bias_add(out, bias, channel_axis)

    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, etab)

    return out

def _convert_separable_convolution(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    if etab.data_layout == 'NHWC':
        kernel_layout = 'HWOI'
    else:
        kernel_layout = 'OIHW'
    weightList = keras_layer.get_weights()
    # depthwise conv
    kernel_h, kernel_w, in_channels, depth_mult = weightList[0].shape
    stride_h, stride_w = keras_layer.strides
    if kernel_layout == 'OIHW':
        weight0 = weightList[0].transpose([2, 3, 0, 1])
    else:
        weight0 = weightList[0]
    params0 = {'weight': etab.new_const(weight0),
               'channels': in_channels * depth_mult,
               'groups': in_channels,
               'kernel_size': [kernel_h, kernel_w],
               'strides': [stride_h, stride_w],
               'dilation': [1, 1],
               'padding': [0, 0],
               'data_layout': etab.data_layout,
               'kernel_layout': kernel_layout}
    if keras_layer.padding == 'valid':
        pass
    # we insert a separate pad operator
    elif keras_layer.padding == 'same':
        in_h = keras_layer.input_shape[1]
        in_w = keras_layer.input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, kernel_w, stride_w)
        params0['padding'] = (pad_t, pad_l, pad_b, pad_r)
    else:
        msg = 'Padding with {} is not supported for operator Separable ' \
              'Convolution in frontend Keras.'
        raise tvm.error.OpAttributeUnImplemented(msg.format(keras_layer.padding))
    depthconv = _op.nn.conv2d(data=inexpr, **params0)
    # pointwise conv
    if kernel_layout == 'OIHW':
        weight1 = weightList[1].transpose([3, 2, 0, 1])
    else:
        weight1 = weightList[1]
        kernel_layout = "HWIO"
    params1 = {'weight': etab.new_const(weight1),
               'channels': weightList[1].shape[3],
               'groups': 1,
               'kernel_size': [1, 1],
               'strides': [1, 1],
               'dilation': [1, 1],
               'data_layout': etab.data_layout,
               'kernel_layout': kernel_layout}
    out = _op.nn.conv2d(data=depthconv, **params1)
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[2])
        if etab.data_layout == 'NCHW':
            out = _op.nn.bias_add(out, bias)
        else:
            out = _op.nn.bias_add(out, bias, axis=-1)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, etab)
    return out


def _convert_flatten(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    # NCHW -> NHWC so that dense can be correctly converted
    if etab.data_layout == 'NCHW':
        inexpr = _op.transpose(inexpr, axes=[0, 2, 3, 1])
    return _op.nn.batch_flatten(inexpr)


def _convert_pooling(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__
    # global pool in keras = global pool + flatten in relay
    global_pool_params = {'layout': etab.data_layout}
    if pool_type == 'GlobalMaxPooling2D':
        return _convert_flatten(
            _op.nn.global_max_pool2d(inexpr, **global_pool_params), keras_layer, etab)
    if pool_type == 'GlobalAveragePooling2D':
        return _convert_flatten(
            _op.nn.global_avg_pool2d(inexpr, **global_pool_params), keras_layer, etab)
    pool_h, pool_w = keras_layer.pool_size
    stride_h, stride_w = keras_layer.strides
    params = {'pool_size': [pool_h, pool_w],
              'strides': [stride_h, stride_w],
              'padding': [0, 0],
              'layout': etab.data_layout}
    if keras_layer.padding == 'valid':
        pass
    elif keras_layer.padding == 'same':
        in_h = keras_layer.input_shape[1]
        in_w = keras_layer.input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, pool_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, pool_w, stride_w)
        params['padding'] = [pad_t, pad_l, pad_b, pad_r]
    else:
        raise tvm.error.OpAttributeUnImplemented(
            'Padding with {} is not supported in operator Pooling.'.format(keras_layer.padding))
    if pool_type == 'MaxPooling2D':
        return _op.nn.max_pool2d(inexpr, **params)
    if pool_type == 'AveragePooling2D':
        params['count_include_pad'] = False
        return _op.nn.avg_pool2d(inexpr, **params)
    raise tvm.error.OpNotImplemented(
        'Operator {} is not supported for frontend Keras.'.format(keras_layer))

def _convert_pooling3d(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__

    if pool_type not in ['MaxPooling3D', 'AveragePooling3D']:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend Keras.'.format(keras_layer))

    pool_d1, pool_d2, pool_d3 = keras_layer.pool_size
    stride_d1, stride_d2, stride_d3 = keras_layer.strides
    params = {'pool_size': [pool_d1, pool_d2, pool_d3],
              'strides': [stride_d1, stride_d2, stride_d3],
              'padding': [0, 0, 0],
              'layout': etab.data_layout}

    if keras_layer.padding == 'valid':
        pass
    elif keras_layer.padding == 'same':
        in_d1 = keras_layer.input_shape[1]
        in_d2 = keras_layer.input_shape[2]
        in_d3 = keras_layer.input_shape[3]
        pad_d1 = _get_pad_pair(in_d1, pool_d1, stride_d1)
        pad_d2 = _get_pad_pair(in_d2, pool_d2, stride_d2)
        pad_d3 = _get_pad_pair(in_d3, pool_d3, stride_d3)
        params['padding'] = [pad_d1[0], pad_d2[0], pad_d3[0], pad_d1[1], pad_d2[1], pad_d3[1]]
    else:
        raise tvm.error.OpAttributeUnImplemented(
            'Padding with {} is not supported in operator Pooling3D.'.format(keras_layer.padding))

    out = _op.transpose(inexpr, axes=(0, 4, 1, 2, 3))
    params['layout'] = "NCDHW"
    if pool_type == 'MaxPooling3D':
        out = _op.nn.max_pool3d(out, **params)
    elif pool_type == 'AveragePooling3D':
        out = _op.nn.avg_pool3d(out, **params)

    return _op.transpose(out, axes=(0, 2, 3, 4, 1))


def _convert_global_pooling3d(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__

    global_pool_params = {'layout': etab.data_layout}
    if pool_type == 'GlobalMaxPooling3D':
        out = _op.nn.global_max_pool3d(inexpr, **global_pool_params)
    elif pool_type == 'GlobalAveragePooling3D':
        out = _op.nn.global_avg_pool3d(inexpr, **global_pool_params)
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend Keras.'.format(keras_layer))

    return _convert_flatten(out, keras_layer, etab)


def _convert_upsample(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    upsample_type = type(keras_layer).__name__
    params = {}
    if upsample_type == 'UpSampling1D':
        h = keras_layer.size
        params['scale_h'] = h
    elif upsample_type == 'UpSampling2D':
        h, w = keras_layer.size
        if h != w:
            raise tvm.error.OpAttributeInvalid(
                'Height must equal width for operator Upsample.')
        params['scale_h'] = h
        params['scale_w'] = h

        if hasattr(keras_layer, 'interpolation'):
            interpolation = keras_layer.interpolation
            if interpolation == 'nearest':
                params['method'] = 'nearest_neighbor'
            else:
                params['method'] = 'bilinear'
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend Keras.'.format(upsample_type))
    params['layout'] = etab.data_layout
    out = _op.nn.upsampling(inexpr, **params)
    return out


def _convert_upsample3d(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    params = {}
    d, h, w = keras_layer.size
    params['scale_d'] = d
    params['scale_h'] = h
    params['scale_w'] = w
    params['layout'] = etab.data_layout
    out = _op.nn.upsampling3d(inexpr, **params)
    return out


def _convert_cropping(inexpr, keras_layer, _):
    _check_data_format(keras_layer)
    crop_type = type(keras_layer).__name__
    if crop_type == 'Cropping2D':
        (_, in_h, in_w, _) = keras_layer.input_shape
        ((crop_t, crop_b), (crop_l, crop_r)) = keras_layer.cropping
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend Keras.'.format(crop_type))
    int32_max = np.iinfo(np.int32).max
    return _op.strided_slice(inexpr, begin=[0, 0, crop_t, crop_l], \
        end=[int32_max, int32_max, in_h-crop_b, in_w-crop_r])


def _convert_batchnorm(inexpr, keras_layer, etab):
    # 1.
    # inexpr value:
    # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198#file-out_head_1500-log-L418-L423

    # 2.
    # keras_layer value:
    # <tensorflow.python.keras.layers.normalization_v2.BatchNormalization object at 0x7f23cf928908>

    # 3.
    # etab value:
    # https://keep.google.com/u/1/#NOTE/1h2_-6AudNGzfWFgddrAYOfT6bIthB72lqXfAaTocKBQmNIvcMfXs0bMf_xn6fQ

    if etab.data_layout == 'NCHW' or len(keras_layer.input_shape) < 4:
        # 进入
        # 1.
        # since etab.data_layout == 'NCHW' == True!
        # but len(keras_layer.input_shape) == 4, so this is False!

        axis = 1
    else:
        axis = 3

    params = {'scale': False,
              'center': False,
              'epsilon': keras_layer.epsilon,
              'axis': axis}
    # 1.
    # params value:
    # {'scale': False, 'center': False, 'epsilon': 1.001e-05, 'axis': 1}

    idx = 0
    if keras_layer.scale:
        # 进入!
        params['scale'] = True
        gamma = keras_layer.get_weights()[idx]
        # 1.
        # idx=0

        # 2.
        # gamma type:
        # ndarray

        # 2.1
        # gamma shape:
        # (64,)

        # 2.2
        # gamma value:
        # https://gist.github.com/shizukanaskytree/d0d99cdcc017dfb9a09997cb6d34ffa1

        params['gamma'] = etab.new_const(gamma)
        idx += 1
        # 1.
        # idx=1
    if keras_layer.center:
        # 进入!
        params['center'] = True
        beta = keras_layer.get_weights()[idx]
        # 1.
        # idx=1

        # 2.
        # beta
        # - type: ndarray;
        # - shape: (64,);
        # - value: https://gist.github.com/shizukanaskytree/d0d99cdcc017dfb9a09997cb6d34ffa1#gistcomment-3340782

        params['beta'] = etab.new_const(beta)
        # 1.
        # # python/tvm/relay/frontend/common.py ↓
        # class ExprTable(object):
        #   def new_const(self, value, shape=None, dtype="float32"):

        idx += 1
        # 1.
        # idx=2

    moving_mean = keras_layer.get_weights()[idx]
    # 1.
    # moving_mean
    # - type: ndarray;
    # - shape: (64,);
    # - value: not interested.

    # 2.
    # idx=2

    moving_var = keras_layer.get_weights()[idx + 1]
    # 1.
    # - type: ndarray;
    # - shape: (64,);
    # - value: not interested.

    # 2.
    # idx+1 == 3

    params['moving_mean'] = etab.new_const(moving_mean)
    params['moving_var'] = etab.new_const(moving_var)
    # in case beta or gamma is not defined
    params['beta'] = etab.new_const(np.zeros(moving_mean.shape)) if \
                     'beta' not in params else params['beta']
    params['gamma'] = etab.new_const(np.ones(moving_mean.shape)) if \
                      'gamma' not in params else params['gamma']
    result, moving_mean, moving_var = _op.nn.batch_norm(inexpr, **params)
    return result


def _convert_padding(inexpr, keras_layer, etab):
  # 1.
  # called from :
  # # python/tvm/relay/frontend/keras.py ↓
  # outs = _convert_map[op_name](inexpr, keras_layer, etab)

  # 2.
  # arguments:
  # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198#file-out_head_1500-log-L389-L398

  # 3.
  # called/used by "ZeroPadding2D" in _convert_map dict.

    _check_data_format(keras_layer)
    # 1.
    # tvm/python/tvm/relay/frontend/keras.py:34: def _check_data_format(keras_layer)
    # 一句话:
    # 本代码不支持 NCHW ❌ 只支持 NHWC ✅

    padding_type = type(keras_layer).__name__
    # 1.
    # keras_layer value:
    # <tensorflow.python.keras.layers.convolutional.ZeroPadding2D object at 0x7f37df479630>

    # 2.
    # type(keras_layer) value:
    # <class 'tensorflow.python.keras.layers.convolutional.ZeroPadding2D'>
    #
    # 2.1
    # within <class 'tensorflow.python.keras.layers.convolutional.ZeroPadding2D'>
    # https://keep.google.com/u/1/#NOTE/19JTtSTw1zMua59CW6zNKN2Yn5vCZ2G7W_G5jFJks2uJKMNeNcQUvnxA8SQfizQ

    # 3.
    # 看到了没, 一个是 object 一个就是 type 了.

    # 4.
    # padding_type value:
    # 'ZeroPadding2D'

    padding = keras_layer.padding
    # 1.
    # padding value:
    # ((3, 3), (3, 3))

    top = left = bottom = right = 0
    if padding_type == 'ZeroPadding2D':
        if isinstance(padding, int):
            top = left = bottom = right = padding
        elif isinstance(padding, tuple):
            if isinstance(padding[0], int):
                top, left = padding
                bottom, right = padding
            elif isinstance(padding[0], tuple):
                # 1.
                # 最后进的是这个分支!!! 其他的不用看了!
                top, bottom = padding[0]
                left, right = padding[1]
            else:
                msg = 'Value {} in attribute "padding" of operator Padding ' \
                      'is not valid.'
                raise tvm.error.OpAttributeInvalid(msg.format(str(padding)))
        else:
            msg = 'Value {} in attribute "padding" of operator Padding is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(str(padding)))
    else:
        msg = 'Operator {} is not supported in frontend Keras.'
        raise tvm.error.OpNotImplemented(msg.format(padding_type))

    if etab.data_layout == 'NCHW':
        # 1.
        # 最后进的是这个分支!!! 其他的不用看了!
        return _op.nn.pad(data=inexpr, pad_width=((0, 0), (0, 0), (top, bottom), (left, right)))
        # 1.
        # inexpr value:
        # {Var} free_var %input_1: Tensor[(1, 3, 224, 224), float32]\n%input_1

        # 2.
        # _op.nn.pad 在哪里?
        # 转入 tvm/relay/op/nn/nn.py
        # def pad(data,
        #         pad_width,
        #         pad_value=0.0,
        #         pad_mode='constant'):
        # 截图课看具体值: https://keep.google.com/u/1/#NOTE/1MAhNFAkoTjiTasbWUKnJG7QS_dpxPJshSSjGjwGY5cnk4sbMpfxUURKUOYN9

    return _op.nn.pad(data=inexpr, pad_width=((0, 0), (top, bottom), (left, right), (0, 0)))
    # 1.
    # 不看, 没进入

def _convert_padding3d(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    padding = keras_layer.padding

    d_pad = h_pad = w_pad = [0, 0]

    # padding can be 'int' or 'tuple of 3 ints' or 'tuple of 3 tuples of 2 ints' or 'tuple
    # of 3 tuples of 2 ints different values'. In all these scenarios keras will send 3
    # tuples of 2 ints.
    if isinstance(padding, tuple) and isinstance(padding[0], tuple):
        d_pad = padding[0]
        h_pad = padding[1]
        w_pad = padding[2]
    else:
        msg = 'Value {} in attribute "padding" of operator ZeroPadding3D is ' \
              'not valid.'
        raise tvm.error.OpAttributeInvalid(msg.format(str(padding)))

    if etab.data_layout == 'NCDHW':
        out = _op.nn.pad(data=inexpr, pad_width=((0, 0), (0, 0),
                                                 (d_pad[0], d_pad[1]),
                                                 (h_pad[0], h_pad[1]),
                                                 (w_pad[0], w_pad[1])))
    else:
        out = _op.nn.pad(data=inexpr, pad_width=((0, 0),
                                                 (d_pad[0], d_pad[1]),
                                                 (h_pad[0], h_pad[1]),
                                                 (w_pad[0], w_pad[1]),
                                                 (0, 0)))
    return out

def _convert_concat(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    if etab.data_layout == 'NHWC' or len(keras_layer.input_shape[0]) < 4:
        axis = -1
    else:
        axis = 1
    return _op.concatenate(_as_list(inexpr), axis=axis)


def _convert_reshape(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    inshape = keras_layer.input_shape # includes batch
    tshape = keras_layer.target_shape # no batch
    if len(inshape) == 3 and len(tshape) == 1:
        # (?, a, b) -> (-1, ab)
        shape = (-1, tshape[0])
    elif len(inshape) in [2, 3] and len(tshape) == 2:
        # (?, cc) -> (-1, c, c)
        # (?, a, b) -> (-1, c, c)
        assert tshape[0] == tshape[1], \
            "Only supports square target shapes, but got {}".format(tshape)
        shape = (-1, ) + tshape
    else:
        # (?, h, w, c) -> (-1, c, H, W)
        # (?, h, w, c) -> (-1, c, hw)
        # (?, hw, c) -> (-1, c, h, w)
        ch = inshape[-1]
        assert ch == tshape[-1], \
            "Only supports last dimension in target shape being equal to " \
            "the channel number of input tensor."
        if etab.data_layout == 'NCHW':
            shape = (-1, ch) + tshape[:-1]
        else:
            shape = (-1,) + tshape[:-1] + (ch,)
    return _op.reshape(inexpr, newshape=shape)


def _convert_lstm(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    if not isinstance(inexpr, list):
        buf = np.zeros((1, keras_layer.units), 'float32')
        c_op = etab.new_const(buf)
        h_op = etab.new_const(buf)
        inexpr = [inexpr, h_op, c_op]
    in_data = inexpr[0]
    next_h = inexpr[1]
    next_c = inexpr[2]
    weightList = keras_layer.get_weights()
    in_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.input_shape)[0])
    kernel_weight = etab.new_const(weightList[0].transpose([1, 0]))
    recurrent_weight = etab.new_const(weightList[1].transpose([1, 0]))
    in_bias = etab.new_const(weightList[2])
    units = list(weightList[0].shape)[1]
    time_steps = in_shape[1]
    in_data = _op.squeeze(in_data, axis=[0])
    in_data = _op.split(in_data, indices_or_sections=time_steps, axis=0)
    # loop for the number of time_steps
    for data in in_data:
        ixh1 = _op.nn.dense(data, kernel_weight, units=units)
        ixh2 = _op.nn.bias_add(_op.nn.dense(next_h, recurrent_weight, units=units), bias=in_bias)
        gate = ixh1 + ixh2
        gates = _op.split(gate, indices_or_sections=4, axis=1)
        in_gate = _convert_recurrent_activation(gates[0], keras_layer)
        in_transform = _convert_recurrent_activation(gates[1], keras_layer)
        next_c = in_transform * next_c + in_gate * _convert_activation(gates[2], keras_layer, None)
        out_gate = _convert_recurrent_activation(gates[3], keras_layer)
        next_h = out_gate * _convert_activation(next_c, keras_layer, None)
    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    out = _op.reshape(next_h, newshape=out_shape)
    return [out, next_h, next_c]


def _convert_simple_rnn(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    if not isinstance(inexpr, list):
        buf = np.zeros((1, keras_layer.units), 'float32')
        prev_op = etab.new_const(buf)
        inexpr = [inexpr, prev_op]
    in_data = inexpr[0]
    prev_op = inexpr[1]
    weightList = keras_layer.get_weights()
    kernel_weight = etab.new_const(weightList[0].transpose([1, 0]))
    recurrent_weight = etab.new_const(weightList[1].transpose([1, 0]))
    in_bias = etab.new_const(weightList[2])
    units = list(weightList[0].shape)[1]
    in_data = _op.nn.batch_flatten(in_data)
    ixh = _op.nn.bias_add(_op.nn.dense(in_data, kernel_weight, units=units), bias=in_bias)
    prev_op = _op.nn.batch_flatten(prev_op)
    ixh2 = _op.nn.dense(prev_op, recurrent_weight, units=units)
    output = ixh + ixh2
    output = _convert_activation(output, keras_layer, None)
    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    output = _op.reshape(output, newshape=out_shape)
    return [output, output]


def _convert_gru(inexpr, keras_layer, etab):
    _check_data_format(keras_layer)
    if not isinstance(inexpr, list):
        buf = np.zeros((1, keras_layer.units), 'float32')
        h_tm1 = etab.new_const(buf)
        inexpr = [inexpr, h_tm1]
    in_data = inexpr[0]
    h_tm1_op = inexpr[1]
    weightList = keras_layer.get_weights()
    kernel_weight = etab.new_const(weightList[0].transpose([1, 0]))
    recurrent_weight = etab.new_const(weightList[1].transpose([1, 0]))
    in_bias = etab.new_const(weightList[2])
    units = list(weightList[0].shape)[1]
    in_data = _op.nn.batch_flatten(in_data)
    matrix_x = _op.nn.bias_add(_op.nn.dense(in_data, kernel_weight, units=units), in_bias)
    # inputs projected by all gate matrices at once
    split_indices = [keras_layer.units, 2 * keras_layer.units]
    gates = _op.split(matrix_x, indices_or_sections=split_indices, axis=1)
    x_z = gates[0]
    x_r = gates[1]
    x_h = gates[2]
    # hidden state projected separately for update/reset and new
    units = 2 * keras_layer.units
    split_indices = [units]
    rec_weights = _op.split(recurrent_weight, indices_or_sections=split_indices, axis=0)
    h_tm1_op = _op.nn.batch_flatten(h_tm1_op)
    matrix_inner = _op.nn.dense(h_tm1_op, rec_weights[0], units=units)
    split_indices = [keras_layer.units]
    recurrent = _op.split(matrix_inner, indices_or_sections=split_indices, axis=1)
    recurrent_z = recurrent[0]
    recurrent_r = recurrent[1]
    rec_act_z = _convert_recurrent_activation(x_z + recurrent_z, keras_layer)
    rec_act_r = _convert_recurrent_activation(x_r + recurrent_r, keras_layer)
    units = keras_layer.units
    recurrent_h = _op.nn.dense(rec_act_r * h_tm1_op, rec_weights[1], units=units)
    act_hh = _convert_activation(x_h + recurrent_h, keras_layer, None)
    # previous and candidate state mixed by update gate
    output = rec_act_z * h_tm1_op + (_expr.const(1., dtype='float32') - rec_act_z) * act_hh
    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    output = _op.reshape(output, newshape=out_shape)
    return [output, output]


def _default_skip(inexpr, keras_layer, _): # pylint: disable=unused-argument
    """Layers that can be skipped because they are train time only."""
    return inexpr

# 1.
# ResNet50 Keras model notations:
# - 📝 TODO;
# - 🙅 Not appear; 自己写 Keras 网络测试吧~
# - ✅ Done!;
# - 👀 Have read but not step debug.
_convert_map = {
    'Dense'                    : _convert_dense, # 📝
    'Activation'               : _convert_activation, # ✅ + 👀
    'Softmax'                  : _convert_advanced_activation, # 📝 🙅‍♂️
    'ReLU'                     : _convert_advanced_activation, # 📝
    'LeakyReLU'                : _convert_advanced_activation, # 📝 🙅
    'PReLU'                    : _convert_advanced_activation, # 📝 🙅
    'ELU'                      : _convert_advanced_activation, # 📝 🙅
    'ThresholdedReLU'          : _convert_advanced_activation, # 📝 🙅

    'AveragePooling2D'         : _convert_pooling, # 📝 🙅
    'MaxPooling2D'             : _convert_pooling, # 📝
    'GlobalAveragePooling2D'   : _convert_pooling, # 📝
    'GlobalMaxPooling2D'       : _convert_pooling, # 📝
    'Conv2D'                   : _convert_convolution, # ✅
    'Conv2DTranspose'          : _convert_convolution, # ✅
    'DepthwiseConv2D'          : _convert_convolution, # ✅
    # 1.
    # What is Depthwise Conv2d?
    # see:
    # - Part 1 — Depthwise Convolution:
    # - Video 1: Iterating 3 kernels through a 3 channel image
    # - Image 6: Depthwise convolution, uses 3 kernels to transform a 12x12x3 image to a 8x8x3 image
    # A Basic Introduction to Separable Convolutions
    # - https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728#:~:text=The%20depthwise%20separable%20convolution%20is,image%20may%20have%20multiple%20channels.
    # comment: 说穿了不值钱系列.

    'SeparableConv2D'          : _convert_separable_convolution, # 📝 🙅

    'Flatten'                  : _convert_flatten,
    'Reshape'                  : _convert_reshape,
    'Concatenate'              : _convert_concat,
    'BatchNormalization'       : _convert_batchnorm, # ✅

    # Specific tf.Keras terminology for batch normalization
    'BatchNormalizationV1'     : _convert_batchnorm, # ✅

    'Add'                      : _convert_merge, # 📝
    'Subtract'                 : _convert_merge, # 📝 🙅
    'Multiply'                 : _convert_merge, # 📝 🙅
    'ZeroPadding2D'            : _convert_padding, # ✅
    # 1.
    # tf ZeroPadding2D API: (一看就懂)
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
    # https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator

    'UpSampling2D'             : _convert_upsample, # 📝 🙅
    # 1.
    # keras UpSampling2D
    # 自己来找例子跑啦
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D

    'Cropping2D'               : _convert_cropping, # 📝 🙅
    # 1.
    # keras Cropping2D
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D

    # 'ZeroPadding1D'          : _convert_padding,
    # 'AveragePooling1D'       : _convert_pooling,
    # 'MaxPooling1D'           : _convert_pooling,
    # 'GlobalAveragePooling1D' : _convert_pooling,
    # 'GlobalMaxPooling1D'     : _convert_pooling,
    # 'Cropping1D'             : _convert_cropping,
    # 'UpSampling1D'           : _convert_upsample,
    # 'Conv1D'                 : _convert_convolution1d,

    'Conv3D'                   : _convert_convolution3d,
    # 'Conv3DTranspose'        : _convert_convolution3d,
    # 'SeparableConv3D'        : _convert_convolution3d,
    'MaxPooling3D'             : _convert_pooling3d,
    'AveragePooling3D'         : _convert_pooling3d,
    'GlobalMaxPooling3D'       : _convert_global_pooling3d,
    'GlobalAveragePooling3D'   : _convert_global_pooling3d,
    'UpSampling3D'             : _convert_upsample3d,
    'ZeroPadding3D'            : _convert_padding3d,

    'SimpleRNN'                : _convert_simple_rnn,
    'LSTM'                     : _convert_lstm,
    'GRU'                      : _convert_gru,
    # 'Bidirectional'          : _convert_bidirectional,
    # 'TimeDistributed'        : _default_skip,

    'Average'                  : _convert_merge,
    'Minimum'                  : _convert_merge,
    'Maximum'                  : _convert_merge,
    'Dot'                      : _convert_merge,
    'Permute'                  : _convert_permute,
    'Embedding'                : _convert_embedding,
    # 'RepeatVector'           : _convert_repeat_vector,

    'InputLayer'               : _default_skip,
    'Dropout'                  : _default_skip,
    'AlphaDropout'             : _default_skip,
    'SpatialDropout2D'         : _default_skip,
    'SpatialDropout1D'         : _default_skip,
    'GaussianDropout'          : _default_skip,
    'GaussianNoise'            : _default_skip,
}


def _check_unsupported_layers(model):
    missing_ops = set()
    for layer in model.layers:
        op_name = type(layer).__name__
        if op_name not in _convert_map:
            missing_ops.add(op_name)

    if missing_ops:
        raise NotImplementedError( \
            "The following operators are not implemented: {}".format(missing_ops))


def keras_op_to_relay(inexpr, keras_layer, outname, etab):
    """Convert a Keras layer to a Relay expression and update the expression table.

    Parameters
    ----------
    inexpr : relay.expr.Expr or a list of it
        The input Relay expression(s).

    keras_layer : keras.layers
        The Keras layer to be converted.

    outname : str
        Name of the output Relay expression.

    etab : relay.frontend.common.ExprTable
        The global expression table to be updated.
    """
    # 1.
    # - inexpr=inexpr
    # - keras_layer=keras_layer
    # - outname=keras_layer.name + ':' + str(node_idx)
    # - etab=etab
    # 上述打印的例子在:
    # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198

    op_name = type(keras_layer).__name__
    # 1.
    #
    if op_name not in _convert_map:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend Keras.'.format(op_name))
    outs = _convert_map[op_name](inexpr, keras_layer, etab)
    # 1.
    # keras 的 _convert_map:
    # https://gist.github.com/shizukanaskytree/2886a73e13c82e1b44cc47aece2fa5d4

    # 2.
    # tf 的 _convert_map:
    # https://gist.github.com/shizukanaskytree/e0cf51f60dae7e28aa6c159f3a7110ce

    # 3.
    # 1. vs 2. : 两者的实现不太一样.

    # 4.
    # inexpr : inline expression
    # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198
    # 上文中的 %1 这样的的式子.

    # 5.
    # outs value:
    # [CallNode(Op(nn.pad), [Var(input_1, ty=TensorType([1, 3, 224, 224], float32))], relay.attrs.PadAttrs(0x55c36734d878), [])]

    # 6.
    # relu case:
    # conv1_relu (Activation)         (None, 112, 112, 64) 0           conv1_bn[0][0]
    # https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5#file-expression_table_names-log-L44
    # so, go to _convert_activation

    # 6.1
    # 那时的结果是:
    # https://gist.github.com/shizukanaskytree/735e0c30df0dbc58e55d7deeef309041
    # - nn.relu(%4)

    outs = _as_list(outs)
    # 1.
    # make outs (intput) as list type

    # 2.
    # outs type:
    # list

    # 3.
    # outs value example:
    # https://gist.github.com/shizukanaskytree/bc1a1247072defbd3cd4c1cebe5cdd17

    # 4.
    # relu outs after _as_list():
    # https://gist.github.com/shizukanaskytree/890d8365c63eba2100e5032d20b0d667
    # https://gist.github.com/shizukanaskytree/735e0c30df0dbc58e55d7deeef309041

    # 4.1
    # 宏观上说其实就是 [CallNode()]

    for t_idx, out in enumerate(outs):
        # 1.
        # 对于这个 for outs 的款项, 结果参考:
        # all result: https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5#file-expression_table_names-log-L44
        # relu result: https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5#file-expression_table_names-log-L457-L458
        #
        # print code: https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5#gistcomment-3340971

        # 2.
        # comment: 奇怪的是只进来一次, 可是那个 outs 是个 nested list , 好多层!
        # https://gist.github.com/shizukanaskytree/890d8365c63eba2100e5032d20b0d667

        # 3.
        # relu out value:
        # https://gist.github.com/shizukanaskytree/4a7f59319bb831006a0feb4b25b039f2#file-etable_name_out-log-L493-L506

        name = outname + ":" + str(t_idx)
        etab.set_expr(name, out)
        # 1.
        # name value:
        # 'conv1_pad:0:0'

        # 2.
        # out value:
        # free_var %input_1: Tensor[(1, 3, 224, 224), float32]
        # nn.pad(%input_1, pad_width=[[0, 0], [0, 0], [3, 3], [3, 3]])

        # 2.1
        # out type:
        # Call

        # 3.
        # etab : relay.frontend.common.ExprTable
        # The global expression table to be updated.

        # 4.
        # # python/tvm/relay/frontend/common.py:296 ↓
        # def set_expr(self, name, expr, force_override=False):

        # 5.
        # ✅ 部分的 (name, out) logging:
        # https://gist.github.com/shizukanaskytree/4a7f59319bb831006a0feb4b25b039f2

        # 6.
        # all name logging:
        # https://gist.github.com/shizukanaskytree/9b37765afb024c948f3714b8a629d7a5


def from_keras(model, shape=None, layout='NCHW'):
    """Convert keras model to relay Function.

    Parameters
    ----------
    model : keras.engine.training.Model or tensorflow.keras.models.Model
        The keras model to be converted.

    shape: dict of str to int list/tuple
        Input shapes of the model, optional

    layout: str
        One of 'NCHW' or 'NHWC', indicates how data should be arranged in
        the output model. Default layout is 'NCHW' as it in general
        performs better across TVM.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by Relay.
    """
    # 1.
    # 使用的 code: go to tutorials/frontend/from_tf_keras.py
    # shape_dict = {'input_1': data.shape}
    # mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)

    # 2.
    # 参数实例化
    # 2.1
    # keras_resnet50: from tensorflow.keras.applications.resnet50 import ResNet50

    # 2.2
    # shape_dict:
    # {'input_1': (1, 3, 224, 224)}

    def _check_model_is_tf_keras():
        return type(model).__module__.startswith("tensorflow.python.keras")

    def _convert_input_layer(keras_layer):
        input_name = keras_layer.name
        # 1.
        # input_name value:
        # input_name: 'input_1'

        input_shape = shape[input_name] if shape is not None and input_name in shape else None
        # 1.
        # shape value:
        # {'input_1': (1, 3, 224, 224)}

        # 2.
        # input_shape value:
        # {tuple: 4} (1,3,224,224)

        etab.set_expr(input_name, new_var(input_name, shape=input_shape))
        # 1.
        # etab : relay.frontend.common.ExprTable
        #   The global expression table to be updated.

        # 2.
        # new_var(input_name, shape=input_shape) 是什么???
        # 函数来自 python/tvm/relay/frontend/common.py : new_var method
        #
        # 2.1
        # python/tvm/relay/expr.py : 里面描述如下
        # Create a new tvm.relay.Var.
        # This is a simple wrapper function that allows specify
        # shape and dtype directly.

        # TODO
        # Get input_name and input_shape. 然后传给 etab.set_expr 这个函数.

        # 3.
        # Q: 如果说研究存在 "input_1" 呢?
        # https://keep.google.com/u/1/#NOTE/1Cx0CMsEHbxyMk-p_Vy8lXuTIBGQgqzFXEJO8NCeuw8UqggRyUEObdJ9HSH_a

    is_tf_keras = _check_model_is_tf_keras()
    # 1.
    # is_tf_keras: True

    if not is_tf_keras:
        # Importing from Keras
        try:
            import keras
        except ImportError:
            raise ImportError("Keras must be installed")
        if keras.backend.backend() != 'tensorflow':
            raise ValueError("Keras frontend currently supports tensorflow backend only.")
        if keras.backend.image_data_format() != 'channels_last':
            raise ValueError("Keras frontend currently supports data_format = channels_last only.")
        expected_model_class = keras.engine.training.Model
        input_layer_class = keras.engine.InputLayer
    else:
        # Importing from Tensorflow Keras (tf.keras)
        try:
            from tensorflow import keras as tf_keras
        except ImportError:
            raise ImportError("Tensorflow must be installed")
        expected_model_class = tf_keras.models.Model
        input_layer_class = tf_keras.layers.InputLayer


    assert isinstance(model, expected_model_class)

    etab = ExprTable()
    # 1.
    # ExprTable() 是什么?
    #
    # etab: <tvm.relay.frontend.common.ExprTable object at 0x7f5fa47011d0>

    # Set global data format.
    assert layout in ['NCHW', 'NHWC', 'NDHWC'], "Layout must be one of 'NCHW', NHWC or NDHWC"
    etab.data_layout = layout
    # 1.
    # tutorials/frontend/from_tf_keras.py 为例
    # layout: NCHW
    # 具体说: (1, 3, 224, 224)

    # 1.
    # 思路解析:
    """
    # python/tvm/relay/frontend/keras.py ↓
    def from_keras(model, shape=None, layout='NCHW'):
      etab = ExprTable()
      etab.data_layout = layout # TODO in MCNN
      # 思想是 BFS + DFS 遍历了所有的 layers! 单枝 DFS, 枝于枝之间的关系是 BFS.
      # 打印的结果: https://gist.github.com/shizukanaskytree/49de0b984c334f8194a68a7e663eb6aa
      # keras model structure graph: https://keep.google.com/u/1/#NOTE/1WTmNAaAFVvzakTZSLQyXOC5ax9ohh2xZMIGuaIZsppvImTzVxl5rffix17jwhw
      for keras_layer in model.layers:
        if isinstance(keras_layer, input_layer_class):
          _convert_input_layer(keras_layer)
        else:
          for node_idx, node in enumerate(inbound_nodes):
            # for..for..for.. 示意图: https://keep.google.com/u/1/#NOTE/1jhfha871akDCWXNKb5q-5W9yY9lOvY-Wb6HXs4P6LmInYzwoUN5gLzPvFxSK
            # 遍历每层 layer 的每个 inbound nodes , 然后是 inbound nodes 的每个 inbound layers
            # **思想是 BFS 遍历了所有的 layers!**
            for n_idx, t_idx, inbound_layer in zip_node:
              if isinstance(inbound_layer, input_layer_class):
                _convert_input_layer(inbound_layer)
              else:
                expr_name = inbound_layer.name + ':' + str(n_idx) + ':' + str(t_idx)
            keras_op_to_relay(inexpr, keras_layer, keras_layer.name + ':' + str(node_idx), etab)
    """
    # https://gist.github.com/shizukanaskytree/49de0b984c334f8194a68a7e663eb6aa

    for keras_layer in model.layers:
        # 1.
        # model.layers 具体是?
        # https://keep.google.com/u/1/#NOTE/1TcPjarIMAxnvAS4_fTmIrhPzox4I0rPfB6IiJVDKlewHjGtcu0hkCpEQAEidyQ
        #
        # 1.1
        # model.layers 具体是? you can copy now !
        # https://gist.github.com/shizukanaskytree/ad09440b0a0c5904b9758114d3a76eff
        # -
        # - tensorflow.python.keras.engine.input_layer.InputLayer
        # - tensorflow.python.keras.layers.convolutional.ZeroPadding2D
        # - tensorflow.python.keras.layers.convolutional.Conv2D
        # - tensorflow.python.keras.layers.normalization_v2.BatchNormalization
        # - tensorflow.python.keras.layers.core.Activation
        # - tensorflow.python.keras.layers.pooling.MaxPooling2D


        # 2.
        # 实例记录
        #
        # keras_layer 其中一个 Conv2D layer 的实例
        # <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x7f5ffd447b38>

        if isinstance(keras_layer, input_layer_class):
            # 第一次 的 for 是进入的, 想想也是.

            # 1.
            # input_layer_class == tf_keras.layers.InputLayer

            # 2.
            # tf_keras.layers.InputLayer 了解下?
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer
            # 点进入看哦.

            _convert_input_layer(keras_layer)
        else:
            inbound_nodes = keras_layer.inbound_nodes if hasattr(keras_layer, 'inbound_nodes') \
                       else keras_layer._inbound_nodes if hasattr(keras_layer, '_inbound_nodes') \
                       else None
            # 1.
            # keras_layer.inbound_nodes 是什么类型? [list]
            # list of "tensorflow.python.keras.engine.node.Node object"
            # 实际打印出来:
            # [<tensorflow.python.keras.engine.node.Node object at 0x7fbac8808978>]

            # 2.
            # tensorflow.python.keras.engine.node.Node 说明:
            # * tf code:
            # - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/node.py

            # 3.
            # inbound 含义是:
            # travelling towards a place rather than leaving it 到达的；入境的
            # travelling towards a particular point
            # 到达的；入境的；归航的；回程的
            # [ADJ 形容词] 归航的;入站的；An inbound flight is one that is arriving from another place. [usu ADJ n]

            # 3.1
            # keras_layer.inbound_nodes 是什么? 图示概念!
            # https://keep.google.com/u/1/#NOTE/1_vGdKUD253w2AYMV5Sw8WnlcLyKSu_PBRVHb4f_rzuHeDaU9Gh4FL7gSh6Iqjg

            if inbound_nodes is None:
                raise TypeError("Unknown layer type or unsupported Keras version : {}"
                                .format(keras_layer))
            for node_idx, node in enumerate(inbound_nodes):
                # 1.
                # enumerate(inbound_nodes):
                # enumerate(list):
                # The enumerate() method adds counter to an iterable and returns it.

                # 2.
                # node type:
                # tensorflow.python.keras.engine.node.Node

                # If some nodes in imported model are not relevant to the current model,
                # skip such layers.
                # - In Keras, model._network_nodes contains keys of all nodes relevant to the
                #   current model;
                # - In tf.Keras, this is already done as part of tensorflow.keras.network.get_config
                if not is_tf_keras and \
                   not model._node_key(keras_layer, node_idx) in model._network_nodes:
                    continue
                inexpr = []
                # 1.
                # inexpr: inline expressions 的意思吧.
                # IR 相关的.

                # 2.
                # inexpr 是干嘛的?
                # A: 如下: inexpr.append(expr)

                # Since Keras allows creating multiple layers from the same name instance,
                # we append node index to the expr name to make it unique.
                # The one exception is InputLayer. Changing input variable names after conversion
                # would confuse users, so we should keep them as far as possible. Fortunately,
                # they are named uniquely to input_1, input_2, input_3... by default.
                zip_node = zip(
                    _as_list(node.node_indices),
                    _as_list(node.tensor_indices),
                    _as_list(node.inbound_layers))
                # 1.
                # node type here is:
                # tensorflow.python.keras.engine.node.Node
                #
                # 说明:
                # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/node.py

                # 2.
                # node.node_indices type type and value:
                # https://keep.google.com/u/1/#NOTE/1W4cv735HI464bd0v8fZc-vCgTAkkgk8H_oqKTn4zMLbRsyPDVcTmnkp53niEyw
                # type: int
                # value: 0

                # 3.
                # node.tensor_indices type and value:
                # type: int
                # value: 0

                # 4.
                # node.inbound_layers type and value:
                # inbound_layers = {InputLayer}
                # 截图:
                # https://keep.google.com/u/1/#NOTE/1yHi6ttM8ydy2ivH4hpnAoxyt2vUPAWtiayRQoyvnZ1gPdFarAW32MJjCInnd

                # 5.
                # 起初我找不到上述的几个变量, 原来他们都是
                # @property
                # def def inbound_layers(self):

                # 6.
                # node_indices 我确实找不到, 搜 tf repo 的代码也搜不到.
                # Node 的所有 member 都在下图中了:
                # https://keep.google.com/u/1/#NOTE/1Tf6UcZUcWxqaTEWzVpNg1H9mOUmZSSaRQIQGZPgSRh78Opw9mWNUM1MBu8j9

                # 6.1
                # node_indices 我确实找不到的 原因:
                # 可能老版本里面是显示地写的:
                # * https://blog.ddlee.cn/posts/4943e1b8/
                # * http://wangbn.blogspot.com/2018/12/keras-node.html
                #
                # 新版本内的都模糊指代了:
                # "call_args=None, call_kwargs=None"
                # complete:
                # class Node:
                #   def __init__(self, layer, call_args=None, call_kwargs=None, outputs=None)

                # 7.
                # Node, Layer 的关系和说明:
                # * [源码笔记]keras源码分析之Layer、Tensor和Node
                # * https://blog.ddlee.cn/posts/4943e1b8/
                #
                # * Keras源码分析(6)：Node
                # * http://wangbn.blogspot.com/2018/12/keras-node.html
                #
                # * Understanding Keras model architecture (node index of nested model)
                # * https://stackoverflow.com/questions/46011749/understanding-keras-model-architecture-node-index-of-nested-model
                #
                # * Keras The Functional API and how to plot keras model.
                # * https://keras.io/guides/functional_api/
                #
                # Node, Layer 的关系和说明 示意图:
                # https://keep.google.com/u/1/#NOTE/1jhfha871akDCWXNKb5q-5W9yY9lOvY-Wb6HXs4P6LmInYzwoUN5gLzPvFxSK

                for n_idx, t_idx, inbound_layer in zip_node:
                    if isinstance(inbound_layer, input_layer_class):
                        # 1.
                        # input_layer_class type ?
                        # {type} <class tensorflow.python.keras.engine.input_layer.InputLayer'>

                        expr_name = inbound_layer.name
                        # 1.
                        # inbound_layer.name value?
                        # "input_1"

                        _convert_input_layer(inbound_layer)
                        # 1.
                        # inbound_layer type:
                        # {type} <class tensorflow.python.keras.engine.input_layer.InputLayer'>

                        # 2.
                        # _convert_input_layer function:
                        # 在用此之前, 已经有了 "input_1" 层了:
                        # https://keep.google.com/u/1/#NOTE/1Cx0CMsEHbxyMk-p_Vy8lXuTIBGQgqzFXEJO8NCeuw8UqggRyUEObdJ9HSH_a

                    else:
                        expr_name = inbound_layer.name + ':' + str(n_idx) + ':' + str(t_idx)
                        # 1.
                        # 打印的 expr_name log
                        # https://gist.github.com/shizukanaskytree/7941abc40cd289f049916dd0f856d759

                    expr = etab.get_expr(expr_name)
                    # 1.
                    # etab type:
                    #

                    # 2.
                    # expr 对于 "input_1" 来说
                    #   free_var %input_1: Tensor[(1, 3, 224, 224), float32]\n%input_1
                    # 对于 "input_1" 来说, 其实例化的值是如下图
                    # https://keep.google.com/u/1/#NOTE/1aNB8_pIvwH2Aj20fYdhBxu_ICR72JWwaFzyjURUVlMAWgTMC0H-mT4hWUKzv

                    # 2.1
                    # free_var
                    # https://docs.tvm.ai/api/python/relay/analysis.html

                    inexpr.append(expr)
                    # 1.
                    # inexpr: inline expressions 的意思吧.

                if len(inexpr) == 1:
                    inexpr = inexpr[0]

                keras_op_to_relay(inexpr, keras_layer, keras_layer.name + ':' + str(node_idx), etab)
                # 1.
                # keras_op_to_relay 函数说明:
                #
                # # python/tvm/relay/frontend/keras.py ↓
                # def keras_op_to_relay(inexpr, keras_layer, outname, etab):
                #     """Convert a Keras layer to a Relay expression and update the expression table.

                # 2.
                # print('\t', '↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ')
                # print('\t', 'inexpr: \n', inexpr)
                # print('\t', '↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ')
                # print('\t', 'keras_layer.name:node_idx', keras_layer.name + ':' + str(node_idx))
                # print('\t', 'etab: ', etab)
                #
                # 对 传入 keras_op_to_relay 的参数比较好奇具体是什么, 如下是打印的:
                # https://gist.github.com/shizukanaskytree/dde75d6f313950042f6ee6e2ab136198

                # 2.1
                # 打印的函数是怎么实现的? 为什么就打印成那个样子呢?
                """
                // src/printer/relay_text_printer.cc:278 ↓
                Doc RelayTextPrinter::PrintExpr(const Expr& expr, bool meta, bool try_inline) {
                  Doc printed_expr;
                  // Printer to print out the IR text format that can be parsed by a parser.
                	if (expr.as<VarNode>()) {
                		// This is our first time visiting the var and we hit the VarNode case
                		// in the visitor. Thus the variable is free.
                		doc_stack_.back() << "free_var " << printed_expr << Doc::NewLine();
                		// Memoization is done in AllocVar.
                		return memo_[expr];
                	}
                }
                """
                #

    # model._output_coordinates contains out_node(oc[0]), node_index(oc[1]) and tensor_index(oc[2])
    # Get all output nodes in etab using the name made from above values.
    # The out exprs were added to etab in keras_op_to_relay using this name.
    outexpr = [etab.get_expr(oc[0].name + ":" + str(oc[1]) + ":" + str(oc[2])) \
               for oc in model._output_coordinates]
    outexpr = outexpr[0] if len(outexpr) == 1 else _expr.Tuple(outexpr)
    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    params = {k:_nd.array(np.array(v, dtype=np.float32)) for k, v in etab.params.items()}
    return IRModule.from_expr(func), params
