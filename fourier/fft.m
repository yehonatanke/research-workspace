function [F, freq_x, freq_y] = compute_2d_fft(I)
[M, N] = size(I);
F = fft2(I);
F = fftshift(F);
freq_x = linspace(-0.5, 0.5, N);
freq_y = linspace(-0.5, 0.5, M);
end

function I_recon = compute_2d_ifft(F)
F = ifftshift(F);
I_recon = real(ifft2(F));
end

function power_spec = compute_power_spectrum(F)
power_spec = abs(F).^2;
end

function phase_spec = compute_phase_spectrum(F)
phase_spec = angle(F);
end

function F_filt = apply_frequency_filter(F, filter_type, cutoff)
[M, N] = size(F);
[X, Y] = meshgrid(linspace(-0.5, 0.5, N), linspace(-0.5, 0.5, M));
D = sqrt(X.^2 + Y.^2);
switch filter_type
    case 'lowpass'
        H = double(D <= cutoff);
    case 'highpass'
        H = double(D >= cutoff);
    case 'bandpass'
        H = double((D >= cutoff(1)) & (D <= cutoff(2)));
    otherwise
        H = ones(M, N);
end
F_filt = F .* H;
end

function [F_log, dynamic_range] = log_transform_fft(F)
F_log = log1p(abs(F));
dynamic_range = [min(F_log(:)), max(F_log(:))];
end

function F_denoised = denoise_frequency(F, threshold)
F_denoised = F;
F_denoised(abs(F) < threshold) = 0;
end

function [freq, psd] = compute_1d_psd(I)
F = fft(I);
psd = abs(F).^2 / length(I);
freq = linspace(0, 0.5, length(I)/2 + 1);
psd = psd(1:length(freq));
end

function F_shift = center_fft(F)
F_shift = fftshift(F);
end

function metrics = compare_fft(F1, F2)
metrics.mse = mean(abs(F1(:) - F2(:)).^2);
metrics.psnr = 20 * log10(max(abs(F1(:))) / sqrt(metrics.mse));
metrics.ssim = ssim(abs(F1), abs(F2));
end
