function [val,img_noise] = QM_function2(img, ref, mask, sigma, noise_coef, noise_coef2, noise_coef3)
    
    load noise_var
%       noise1 = rand(size(img))-0.5;
%       noise2 = rand(size(img))-0.5;
%       noise3 = rand(size(img))-0.5;
% noise1 = imgaussfilt3(noise1,0.5); % smooth noise 
% noise1 = 0.5 * ( noise1 ./ max( abs(noise1(:)) ) );
% noise2 = imgaussfilt3(noise2,1); % smooth noise 
% noise2 = 0.5 * ( noise2 ./ max( abs(noise2(:)) ) );
% noise3 = imgaussfilt3(noise3,2); % smooth noise 
% noise3 = 0.5 * ( noise3 ./ max( abs(noise3(:)) ) );
% figure
% subplot(1,3,1)
% imagesc(squeeze(noise1(:,:,19)))
% subplot(1,3,2)
% imagesc(squeeze(noise2(:,:,19)))
% subplot(1,3,3)
% imagesc(squeeze(noise3(:,:,19)))
    
    % fake mask
    if mask == 1
        mask = (img .* ref)>0;
    end
    
    % gaussian smoothing
    if sigma < 0.1
        sigma = 0.1;
    end
    %img_gauss = imgaussfilt3(img,sigma);
    %img_gauss = img_gauss .* double(mask);
    img_gauss = img .* double(mask);
    
    % add in noise
    noise_range = noise_coef * std(double(img_gauss(find(img_gauss))));
    noise_add = noise_range * noise1;
    noise_add = imgaussfilt3(noise_add,0.5); % smooth noise 
    img_noise = img_gauss + noise_add;
    
    % smooth image
    img_gauss = imgaussfilt3(img_noise,sigma);
    img_noise = img_gauss .* double(mask);
    
%    val = sum((ref(:)-img_gauss(:)).^2)/numel(ref);
    
    % reapply mask
    img_noise = img_noise .* double(mask);

    % add in noise
    noise_range = noise_coef2 * std(double(img_noise(find(img_noise))));
    noise_add = noise_range * noise2;
    noise_add = imgaussfilt3(noise_add,noise_coef3); % smooth noise 
    img_noise = img_noise + noise_add;
    
    % reapply mask
    img_noise = img_noise .* double(mask);
    
%     % add in noise
%     noise_range = noise_coef3 * std(double(img_noise(find(img_noise))));
%     noise_add = noise_range * noise3;
%     noise_add = imgaussfilt3(noise_add,2); % smooth noise 
%     img_noise = img_noise + noise_add;
    
    % reapply mask
    img_noise = img_noise .* double(mask);
    img_noise = img_noise - min(img_noise(:));
    img_noise = img_noise .* double(mask);
    img_noise = img_noise./max(img_noise(:));
    
    % renormalize image
    idx = find(mask);
%     img_clean = zeros(size(mask));
%     img_clean = img_noise - min(img_noise(:));
%     img_clean = img_clean .* mask;
%     img_noise = img_clean;
%     img_noise = img_noise./max(img_noise(:));
    
%     my_corr = corrcoef(img_noise(idx),ref(idx));
%     if  my_corr(2) < 0
%         val = 1;
%     else
%         val = 1 - my_corr(2);
%     end

    R_orig = corrcoef(img(idx),ref(idx));
    R_new = corrcoef(img_noise(idx),ref(idx));
    R = R_orig(2) - R_new(2);
    %val = abs(mean(img_noise(idx)) - mean(ref(idx))) + abs(std(img_noise(idx)) - std(ref(idx)));
    %val = abs((kurtosis(img_noise(idx))-3) - (kurtosis(ref(idx))-3)) + abs(skewness(img_noise(idx)) - skewness(ref(idx))) + abs(std(img_noise(idx)) - mean(std(ref(idx))));
    
    val = abs(mean(img_noise(idx)) - mean(ref(idx)))/2 + abs(skewness(img_noise(idx)) - skewness(ref(idx)))/10 + abs(std(img_noise(idx)) - mean(std(ref(idx))))/2 + R;
    
    [val abs(mean(img_noise(idx)) - mean(ref(idx)))/2 abs(skewness(img_noise(idx)) - skewness(ref(idx)))/10 abs(std(img_noise(idx)) - mean(std(ref(idx)))) R]
    [sigma, noise_coef, noise_coef2, noise_coef3]
    %abs(mean(img_noise(idx)) - mean(ref(idx)))/2 + abs(skewness(img_noise(idx)) - skewness(ref(idx)))/5 + abs(std(img_noise(idx)) - mean(std(ref(idx))));
    %val = abs(std(img_noise(idx)) - std(ref(idx)));
    
    
end