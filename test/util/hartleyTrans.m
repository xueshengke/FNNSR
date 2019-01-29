function output = hartleyTrans(input, method)
%% hartley transform implemented by fft
% method = 't': forward computation
% method = 'i': inverse computation
insize = size(input);
factor = 1 ./ sqrt(insize(1)*insize(2));

for i = 1 : size(input, 3)
    if isequal(method, 't')
        f= fftshift(fft2(input(:,:,i)) * factor);
    else
        f= fft2(ifftshift(input(:,:,i))) * factor;
    end
    output(:,:,i) = real(f) - imag(f);
end

end