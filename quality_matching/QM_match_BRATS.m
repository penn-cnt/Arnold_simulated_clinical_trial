% add scripts to path
addpath('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\scripts');
load('QM_var.mat'); % values used for image transformation

% get subject for processing
cd('D:\MICCAI_BraTS_2019_Data_Training\LGG'); % To process HGG, change path
subs = dir;
subs = subs(3:end);

% read in HF image for resizing images
HF = dir('C:\Users\tca11\Desktop\CNT\Projects\Hyperfine\data\P001\HF_BrainExtractionBrain.nii.gz');
HF = niftiread(fullfile(HF.folder,HF.name));
HF = double(HF); % convert to floating point
HF = HF ./ max(HF(:)); % normalize to one

% make output directory
outdir = 'D:\BraTS_2019_HF_like\';
mkdir(outdir);

% loop through each subject
for sub = 1:length(subs)
    
    % read in flair image
    folder = fullfile(subs(sub).folder,subs(sub).name);
    path_flair = dir([folder,'\*_flair.nii.gz']);
    path_flair = fullfile(path_flair(1).folder,path_flair(1).name);
    flair = niftiread(path_flair);
    
    % read in segmentation
    path_seg = dir([folder,'\*_seg.nii.gz']);
    path_seg = fullfile(path_seg(1).folder,path_seg(1).name);
    seg = niftiread(path_seg); 
    
    % resize images and normalize intensity of image
    flair = flair(:,end:-1:1,:); % flip to match HF orientation
    flair = flair(40:200,20:200,:); % decrease boarder
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
    
    % match sizing for segmentation
    seg = seg(:,end:-1:1,:); % flip to match HF orientation
    seg = seg(40:200,20:200,:); % decrease boarder
    seg = imresize3(seg,size(HF)); % resize to HF
    
    ref = ones(size(flair)); % provide mask that includes all voxels
    [val,img_noise] = QM_function(flair,ref,1,vals(1),vals(2),vals(3),vals(4));
    
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

%% original BraTS data

% make output directory
outdir = 'D:\BraTS_2019\';
mkdir(outdir);

% loop through each subject
for sub = 1:length(subs)
    
    % read in flair image
    folder = fullfile(subs(sub).folder,subs(sub).name);
    path_flair = dir([folder,'\*_flair.nii.gz']);
    path_flair = fullfile(path_flair(1).folder,path_flair(1).name);
    flair = niftiread(path_flair);
    
    % read in segmentation
    path_seg = dir([folder,'\*_seg.nii.gz']);
    path_seg = fullfile(path_seg(1).folder,path_seg(1).name);
    seg = niftiread(path_seg); 
    
    % resize images and normalize intensity of image
    flair = flair(:,end:-1:1,:); % flip to match HF orientation
    flair = flair(40:200,20:200,:); % decrease boarder
    flair = double(flair); % convert to floating point
    flair = flair ./ max(flair(:)); % normalize to one
    
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
    
    % match sizing for segmentation
    seg = seg(:,end:-1:1,:); % flip to match HF orientation
    seg = seg(40:200,20:200,:); % decrease boarder
    
    % add mask to reduce noise outside of iamge
    flair = flair .* mask;
    
    % save out slices into appropriate folders
    mkdir(fullfile(outdir,'LGG',subs(sub).name));
    mkdir(fullfile(outdir,'LGG',subs(sub).name,'1'));
    mkdir(fullfile(outdir,'LGG',subs(sub).name,'0'));
    for i = 1:size(flair,3)
        cur_slice = squeeze(flair(:,:,i));
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