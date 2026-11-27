r"""Plan M11 (Step F): export the learned detectors to ONNX, quantise them to int8.

    LSTM autoencoder   exported with its scoring step inside the graph: the output is
                       the window score (mean squared reconstruction error), so the
                       deployed file does the whole detection for one window
    supervised LSTM    exported as it is: the output is the window's attack logit
    tree models        the whole scikit-learn pipeline (median imputer, scaler,
                       forest) through skl2onnx; the output is the attack probability

The LSTMs go through PyTorch's TorchScript exporter, which writes ONNX's own LSTM
operator; ONNX Runtime's dynamic quantisation turns that into DynamicQuantizeLSTM
and every Linear layer into an int8 matrix product. "Dynamic" means the weights are
stored as int8 and the activations are quantised on the fly for each window, so no
calibration data is needed.

A tree ensemble has no weight matrices: its file is split features, thresholds and
leaf values, and ONNX Runtime's quantiser finds nothing to convert (it refuses the
model). The forest is therefore reported in full precision only, and Step F measures
smaller forests instead (benchmark.py).
"""

import contextlib
import io
import warnings

import numpy as np

# Dynamic int8 settings that convert every LSTM and Linear layer. benchmark.py picks
# one on validation windows; the test flights never take part in the choice.
INT8_SETTINGS = {
    "default": {},
    "reduce_range": {"reduce_range": True},
    "per_channel": {"per_channel": True},
    "per_channel_reduce_range": {"per_channel": True, "reduce_range": True},
    "uint8_weights": {"weight_type": "QUInt8"},
}


def _quiet():
    """Silence the exporters' console output (it carries characters a Windows
    console cannot print)."""
    stack = contextlib.ExitStack()
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter("ignore")
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    return stack


def ae_scorer(model):
    """The autoencoder with Module 07's window score as its output."""
    import torch

    class Scored(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return ((self.m(x) - x) ** 2).mean(dim=1).mean(dim=1)

    return Scored(model).eval()


def export_lstm(kind, model, path, example):
    """Write one LSTM to ONNX with a free batch size. example: (1, T, k) float32."""
    import torch

    net = ae_scorer(model) if kind == "ae" else model.eval()
    tmp = path.with_name(path.stem + ".partial.onnx")
    with _quiet(), torch.no_grad():
        torch.onnx.export(net, (torch.from_numpy(example),), str(tmp),
                          input_names=["x"], output_names=["score"],
                          dynamic_axes={"x": {0: "batch"}, "score": {0: "batch"}},
                          dynamo=False, opset_version=17)  # fmt: skip
    tmp.replace(path)
    return path


def export_forest(pipe, n_features, path):
    """Write a scikit-learn pipeline (imputer, scaler, tree model) to ONNX."""
    from skl2onnx import to_onnx

    model = pipe[-1]
    with _quiet():
        onx = to_onnx(pipe, np.zeros((1, n_features), np.float32),
                      options={id(model): {"zipmap": False}},
                      target_opset={"": 17, "ai.onnx.ml": 3})  # fmt: skip
    tmp = path.with_name(path.stem + ".partial.onnx")
    tmp.write_bytes(onx.SerializeToString())
    tmp.replace(path)
    return path


def quantize(src, dst, setting):
    """Dynamic int8 quantisation. Returns None, or the quantiser's refusal message."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    kw = dict(INT8_SETTINGS[setting])
    kw["weight_type"] = getattr(QuantType, kw.get("weight_type", "QInt8"))
    tmp = dst.with_name(dst.stem + ".partial.onnx")
    try:
        with _quiet():
            quantize_dynamic(str(src), str(tmp), **kw)
    except ValueError as err:  # e.g. a tree ensemble: no ai.onnx operators to convert
        tmp.unlink(missing_ok=True)
        return str(err)
    tmp.replace(dst)
    return None


def session(path, threads=1):
    """An ONNX Runtime CPU session on `threads` threads, operators run in order."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def scorer(sess):
    """fn(x) -> one attack score per row of x, for an LSTM or a tree pipeline."""
    name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    if "probabilities" in outs:
        return lambda x: sess.run(["probabilities"], {name: x})[0][:, 1]
    return lambda x: sess.run(None, {name: x})[0].reshape(-1)


def score_all(sess, x, batch=4096):
    fn = scorer(sess)
    parts = [fn(np.ascontiguousarray(x[i : i + batch], np.float32))
             for i in range(0, len(x), batch)]  # fmt: skip
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)
