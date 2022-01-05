% add scripts to path
addpath('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\scripts');
load('QM_var.mat'); % values used for image transformation

% get subject for processing
cd('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\data\normal_flairs');
subs = dir;
subs = subs(3:end);

% read in HF image for resizing images
HF = dir('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\tmp\P001_clinical\HF_BrainExtractionBrain.nii.gz');
HF = niftiread(fullfile(HF.folder,HF.name));
HF = double(HF); % convert to floating point
HF = HF ./ max(HF(:)); % normalize to one

% make output directory
outdir = 'D:\normal_HF_like\';
mkdir(outdir);

% loop through each subject
for sub = 1:length(subs)
    
    % read in flair image
    if intersect(sub,[1,2,4])
        file_suffix = '\scans\anat2-FLAIR\resources\NIFTI\files\';
        folder = fullfile(subs(sub).folder,subs(sub).name,file_suffix);
    else
        file_suffix = '\scans\anat3-FLAIR\resources\NIFTI\files\';
        folder = fullfile(subs(sub).folder,subs(sub).name,file_suffix);
    end
    path_flair = dir([folder,'\*FLAIR.nii*']);
    path_flair = fullfile(path_flair(1).folder,path_flair(1).name);
    flair = niftiread(path_flair); 
    
    % resize images and normalize intensity of image
    flair = double(flair); % convert to floating point
    flair = flair ./ max(flair(:)); % normalize to one
    flair = imresize3(flair,size(HF)); % resize to HF
    
    % generate mask of brain and skull
    mask = imgaussfilt(flair,3);
    mask = mask > 0.05;
    CC = bwconncomp(mask); % get clusters
    CC_length = [];
    for i = 1:length(CC.PixelIdxList)
        CC_length(i) = length(CC.PixelIdxList{i}); % get cluster sizes
    end
    [voxels,idx] = max(CC_length);
    mask = zeros(size(flair));
    mask(CC.PixelIdxList{idx}) = 1; % mask of only largest cluster
    mask = imfill(mask); % fill in holes in mask
    
    ref = ones(size(flair)); % provide mask that includes all voxels
    [val,img_noise] = QM_function(flair,ref,1,vals(1),vals(2),vals(3),vals(4));
    
    % add mask to reduce noise outside of iamge
    img_noise = img_noise .* mask;
    
    % save out slices into appropriate folders
    mkdir(fullfile(outdir,['sub',num2str(sub)]));
    for i = 1:size(img_noise,3)
        cur_slice = squeeze(img_noise(:,:,i));
        filename = fullfile(outdir,['sub',num2str(sub)],['slice',num2str(i),'.png']);
        imwrite(cur_slice,filename)
    end
    
end