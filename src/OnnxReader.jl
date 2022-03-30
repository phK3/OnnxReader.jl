module OnnxReader

using PyCall

const onnx_reader = PyNULL()

function __init__()
    # current directory is not added to path by default, so add it here
    pushfirst!(PyVector(pyimport("sys")["path"]), @__DIR__)
    
    copy!(onnx_reader, pyimport("onnx_reader"))
end

# just interface to onnx_reader.py 
function load_network(filename)
    or = onnx_reader.OnnxReader(filename)
    or.load_onnx_network()
    ws, bs = or.get_weights_and_biases()

    return ws, bs
end

export load_network

end # module
