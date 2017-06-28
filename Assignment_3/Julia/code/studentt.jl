
function studentt(d, sigma::Float64, alpha::Float64)
    p = (((d.^2)./(2*sigma^2)).+1).^(-alpha);
    return p;
end

function grad_studentt(d, sigma::Float64, alpha::Float64)
    g = ((-alpha).*(d./(sigma^2))).*(((d.^2)./(2*sigma^2).+1).^(-alpha-1));
    return g;
end

function grad_studentt_sigma(x, sigma::Float64, alpha::Float64)
  gs = (2 * alpha * 4 * sigma .* x) ./ (2 * sigma^2 .+ x.^2).^2;
  return gs;
end

function grad_studentt_alpha(x, sigma::Float64, alpha::Float64)
  ga = -2 .* x ./ (2 * sigma ^2 .+ x.^2);
  return ga;
end
