Simple reader for (non-fancy) neural networks in `.onnx` format for Julia.

The currently supported operations are: `Add, Sub, MatMul, Gemm, Relu`. (`Flatten` is not supported, but is ignored as long as only one dimension is larger than one.)

The code is based on the python implementation in [NNENUM](https://github.com/stanleybak/nnenum/blob/master/src/nnenum/onnx_network.py) and just uses `PyCall.jl` to interface to python. 

# Installation

Type `]` in the Julia REPL to activate the package manager. The OnnxReader can then be installed via 
```julia
(v1.0) pkg> add https://github.com/phK3/OnnxReader.jl
```
(Make sure `onnx` and `numpy` are available in the Python environment used by `PyCall.jl`)

# Example

```julia
using OnnxReader

ws, bs = load_network("./example_nns/ACASXU_run2a_1_1_batch_2000.onnx")
@show size(ws)
@show size(bs)
```

Should return the weights and biases of the respective network.


