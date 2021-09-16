import torch


def qsgd(vec, n_bits=3, frob_norm=False):
    """
    Implementation of random compresson using QSGD. 
    Ref -> Alistarh, Dan, et al. "QSGD: Communication-efficient SGD via gradient quantization and encoding." Advances in Neural Information Processing Systems 30 (2017): 1709-1720.
    
    Original implementation used frobenius norm. Here we found that max norm is better in for GCCA case.
    """
    if vec.is_cuda:
        vec = vec.to('cpu')
    s = 2**(n_bits-1) - 1
    abs_vec = torch.abs(vec)
    if frob_norm:
        norm = torch.norm(vec, 'fro')
    else:
        norm = torch.max(torch.abs(vec))
    normalized_vec = abs_vec/norm
    scaled_vec = normalized_vec * s
    l = torch.floor(scaled_vec)
    prob = scaled_vec - l

    # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
    probabilities = scaled_vec - l
    r = torch.rand(l.size())
    l += (probabilities > r)
    signs = torch.sign(vec)

    signed_l = l*signs
    quantized = signed_l*norm/s
    return quantized

def deterministic_quantize(vec, n_bits=3):
    """
    Deterministic symmetric qunatizer
    """
    s = 2**(n_bits-1) - 1
    max_val = vec.abs().max()
    return ((s/max_val)*diff).round()*(max_val/s)

if __name__=='__main__':    
    n_bits = 8
    vec = torch.randn(10,15)
    frob_norm = False
    compressed = qsgd(vec, n_bits=n_bits, frob_norm = frob_norm)
    print(torch.norm(compressed-vec))
