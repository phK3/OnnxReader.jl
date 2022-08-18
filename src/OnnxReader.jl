module OnnxReader

using PyCall

const onnx_reader = PyNULL()

function __init__()
    # current directory is not added to path by default, so add it here
    pushfirst!(PyVector(pyimport("sys")["path"]), @__DIR__)
    
    copy!(onnx_reader, pyimport("onnx_reader"))
end


# sometimes the python onnx reader already returns a julia Matrix?
function convert2mat(w::Matrix; dtype=AbstractFloat)
    return convert.(dtype, w)
end


# if the python onnx reader returns PyObject, we need to cast it to julia Matrix
function convert2mat(w; dtype=AbstractFloat)
    w = convert.(Vector{dtype}, w)  # w is now vec of vecs
    w = convert(Matrix, reduce(hcat, w)')  # w is now matrix
    return w
end


# we want all biases to be vectors, don't want single numbers
function convert2vec(b; dtype=AbstractFloat)
    b = convert.(dtype, b)
    if length(b) == 1
        b = [b]
    end
    
    return b    
end


# just interface to onnx_reader.py 
function load_network(filename; dtype=AbstractFloat)
    or = onnx_reader.OnnxReader(filename)
    or.load_onnx_network()
    ws, bs = or.get_weights_and_biases()

    ws = [convert2mat(w, dtype=dtype) for w in ws]
    bs = [convert2vec(b, dtype=dtype) for b in bs]

    return ws, bs
end

export load_network

end # module
