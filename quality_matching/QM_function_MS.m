function [val,img_noise] = QM_function_MS(img, ref, mask, sigma, noise_coef, noise_coef2, noise_coef3)
    
    % generate mask containing entire image if one is not provided
    if mask == 1
        mask = (img .* ref) > 0;
    end
    
    % set lower bound for gaussian smoothing parameter (must be non-zero)
    if sigma < 0.1
        sigma = 0.1;
    end
    img_gauss = img .* double(mask); % mask image if mask is provided
    
    % generate random noise
    noise1 = rand(size(img))-0.5;
    noise2 = rand(size(img))-0.5;
    noise3 = rand(size(img))-0.5;
    
    % add in noise
    noise_range = noise_coef * std(double(img_gauss(find(img_gauss)))); % weight noise based on image variability
    noise_add = noise_range * noise1; % generate noise to add to image
    noise_add = imgaussfilt3(noise_add,0.5); % smooth noise 
    img_noise = img_gauss + noise_add; % add noise to image
    
    % smooth image
    img_gauss = imgaussfilt3(img_noise,sigma);
    img_noise = img_gauss .* double(mask);

    % add additional noise
    noise_range = noise_coef2 * std(double(img_noise(find(img_noise)))); % weight noise based on image variability
    noise_add = noise_range * noise2; % generate noise to add to image
    noise_add = imgaussfilt3(noise_add,noise_coef3); % smooth noise 
    img_noise = img_noise + noise_add; % add noise to image
    
    % reapply mask
    img_noise = img_noise .* double(mask);
        
    % normailze image range
    img_noise = img_noise - min(img_noise(:));
    img_noise = img_noise .* double(mask);
    img_noise = img_noise./max(img_noise(:));
    idx = find(mask);
    
    % compare original image, reference image, and new image
    R = corr(img(idx),ref(idx)) - corr(img_noise(idx),ref(idx));
    val = abs(mean(img_noise(idx)) - mean(ref(idx)))/2 + abs(skewness(img_noise(idx)) - skewness(ref(idx)))/10 + abs(std(img_noise(idx)) - mean(std(ref(idx))))/2 + R;
    [val abs(mean(img_noise(idx)) - mean(ref(idx)))/2 abs(skewness(img_noise(idx)) - skewness(ref(idx)))/10 abs(std(img_noise(idx)) - mean(std(ref(idx)))) R]
    
end