
function studentt(d, sigma::Float64, alpha::Float64)
    p = (( (d.^2) ./ (2 * sigma^2) ) .+ 1).^(-alpha)
    return p;
end

function grad_studentt(d, sigma::Float64, alpha::Float64)
    p = ( (-alpha) .* ( d ./ (sigma^2) ) ) .* ( ( (d.^2) ./ (2 * sigma^2) .+ 1) .^ (-alpha-1) )
    return p;
end
