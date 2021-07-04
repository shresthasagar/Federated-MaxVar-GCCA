
function quantized = qsgd(V, n_bits)
    % compress
    % norm_V = norm(V, 'fro');
    norm_V = max(abs(V), [], 'all');

    V_normalized = abs(V)/norm_V;
    s = 2^n_bits;
    V_scaled = V_normalized * s;
    l = 0;
    prob = V_scaled - l;
    
    r = rand(size(V));
    l = l + 1*(prob>r);
    V_sign = sign(V);

    V_recovered = l.*V_sign;
    quantized =  V_recovered*norm_V / s;

end

function y = bound(x,bl,bu)
    % return bounded value clipped between bl and bu
    y=min(max(x,bl),bu);
end