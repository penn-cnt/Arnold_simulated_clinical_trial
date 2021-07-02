vals = [0.5000    2.5000    1.5000    0.5000]; % values used for image transformation

addpath('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\scripts');

% get subject for processing
subs = dir('D:\MICCAI_BraTS_2019_Data_Training\LGG');
subs = subs(3:end);

% read in HF image for resizing images
HF = dir('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\tmp\P001_clinical\HF_BrainExtractionBrain.nii.gz');
HF = niftiread(fullfile(HF.folder,HF.name));
HF = double(HF); % convert to floating point
HF = HF ./ max(HF(:)); % normalize to one

% make output directory
outdir = 'D:\BraTS_intensity_matched\';
mkdir(outdir);

for sub = 1:length(subs)
    
    for intensity_percent = 0:20:100
        
        % load in image data
        % read in flair image
        folder = fullfile(subs(sub).folder,subs(sub).name);
        path_flair = dir([folder,'\*_flair.nii.gz']);
        path_flair = fullfile(path_flair(1).folder,path_flair(1).name);
        flair = niftiread(path_flair);
        
        flair = double(flair);
        flair = flair./max(flair(:));
        
        % read in segmentation
        path_seg = dir([folder,'\*_seg.nii.gz']);
        path_seg = fullfile(path_seg(1).folder,path_seg(1).name);
        seg = niftiread(path_seg);
        
        % get slice with highest label count
        [~,slice] = max(squeeze(mean(mean(seg))));
        slice = slice(1); % incase max occurs multiple times
        
        % get ratio gradient
        idx1 = find(seg==1);
        idx2 = find(seg==2);
        idx4 = find(seg==4);
        ratio1 = mean(double(flair(find(flair)))) / mean(double(flair(idx1)));
        ratio2 = mean(double(flair(find(flair)))) / mean(double(flair(idx2)));
        ratio4 = mean(double(flair(find(flair)))) / mean(double(flair(idx4)));
        my_ratio = ones(size(seg));
        my_ratio(idx1) = ratio1;
        my_ratio(idx2) = ratio2;
        my_ratio(idx4) = ratio4;
        my_ratio = imgaussfilt3(my_ratio,1);
        ratio_diff=ones(size(my_ratio))-my_ratio;  % used to modulate intensity
        
        % make output directory
        outdir = ['D:\BraTS_intensity_matched\p',num2str(intensity_percent),'\'];
        mkdir(outdir);
        
        % get ratio of intensity between tissue types
        ratio_current = my_ratio + ( ratio_diff .* (intensity_percent/100) );
        
        % apply gradient to image
        flair_mod = double(flair) .* ratio_current;
        flair_mod_orig = flair_mod;
        flair_smooth = imgaussfilt3(flair_mod,0.75);
        se = strel('sphere',3);
        seg_dilated = imdilate(seg,se);
        seg_erode = imerode(seg,se);
        seg_diff = (seg_dilated>0) - (seg_erode>0);
        flair_mod(find(seg_diff)) = flair_smooth(find(seg_diff));
        flair_mod(find(seg_erode)) = flair_mod_orig(find(seg_erode));
        
        % resize images and normalize intensity of image
        flair_mod = flair_mod(:,end:-1:1,:); % flip to match HF orientation
        flair_mod = flair_mod(40:200,20:200,:); % decrease boarder
        flair_mod = double(flair_mod); % convert to floating point
        flair_mod = flair_mod ./ max(flair_mod(:)); % normalize to one
        flair_mod = imresize3(flair_mod,size(HF)); % resize to HF
        
        % generate mask of brain and skull
        mask = imgaussfilt(flair_mod,3);
        mask = mask > 0.05;
        CC = bwconncomp(mask); % get clusters
        CC_length = [];
        for i = 1:length(CC.PixelIdxList)
            CC_length(i) = length(CC.PixelIdxList{i}); % get cluster sizes
        end
        [voxels,idx] = max(CC_length);
        mask = zeros(size(flair_mod));
        mask(CC.PixelIdxList{idx}) = 1; % mask of only largest cluster
        mask = imfill(mask); % fill in holes in mask
        
        % match sizing for segmentation
        seg = seg(:,end:-1:1,:); % flip to match HF orientation
        seg = seg(40:200,20:200,:); % decrease boarder
        seg = imresize3(seg,size(HF)); % resize to HF
        
        ref = ones(size(flair_mod)); % provide mask that includes all voxels
        [val,img_noise] = QM_function2(flair_mod,ref,1,vals(1),vals(2),vals(3),vals(4));
        
        % add mask to reduce noise outside of iamge
        img_noise = img_noise .* mask;
        
        % save out slices into appropriate folders
        mkdir(fullfile(outdir,'LGG',subs(sub).name));
        mkdir(fullfile(outdir,'LGG',subs(sub).name,'1'));
        mkdir(fullfile(outdir,'LGG',subs(sub).name,'0'));
        for i = 1:size(img_noise,3)
            cur_slice = squeeze(img_noise(:,:,i));
            cur_seg = squeeze(seg(:,:,i));
            if sum( cur_seg(:) ) >= 1
                filename = fullfile(outdir,'LGG',subs(sub).name,'1',['slice',num2str(i),'_lesion.png']);
                imwrite(cur_slice,filename)
            else
                filename = fullfile(outdir,'LGG',subs(sub).name,'0',['slice',num2str(i),'_nonlesion.png']);
                imwrite(cur_slice,filename)
            end
        end
        
    end
    
    
end
