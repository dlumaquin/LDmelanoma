%% LD analysis script

clear all
close all

% if czis are saved in a folder (leave empty otherwise):
path_to_folder = %insertpath;


% if multiple images are stored within 1 czi file (leave empty otherwise):
path_to_czi = {};

save_MIPs = 1;
n_channels = 4;


%% Import CZI file and save each series

ims_405 = {};
ims_488 = {};
ims_555 = {};
if n_channels == 4
    ims_647 = {};
end

show_ims = 0;
im_name_list = {};

% for multiple .czi files stored in 1 folder
if ~isempty(path_to_folder) 

    % find image files within folder
    im_folder = path_to_folder{1};
    im_files = dir(fullfile(im_folder, '*.czi'));
    im_paths = fullfile(im_folder, {im_files.name});
    
    experiment_name = extractAfter(im_folder, 'RPMI/'); % the part of the path to your folder that precedes the image names
    experiment_name = experiment_name(1:end-1);
    
    for ii = 1:length(im_paths)
        
        % waitbar
        if ii == 1
            f = waitbar((ii-1)/length(im_paths), append('Processing image ', string(ii), ' of ', string(length(im_paths)), '...'));
        else
            waitbar((ii-1)/length(im_paths), f, append('Processing image ', string(ii), ' of ', string(length(im_paths)), '...'));
        end
        
        file_import = im_paths{ii};
        data = bfopen(file_import);
        
        % images within data are stored in this format:
        % each row corresponds to 1 series (containing all channels)
        % within each element of the first column (1,1 2,1 3,1 etc) there is an image file that corresponds to each individual image within the stack
        % images are stored as blue image (z-stack position 1), green image (z-stack position 1), red image (z-stack position 1), blue image (z-stack position 2) etc
        
        % determine number of images within the series
        [n_series,~] = size(data);
        
        for jj = 1:n_series
            
            series = data{jj,1};
            series_names = series(:,2);
            
            % find the images corresponding to each channel
            if n_channels == 3
                idx_405 = find(~cellfun(@isempty,(strfind(series_names, 'C=1/3'))));
                idx_488 = find(~cellfun(@isempty,(strfind(series_names, 'C=2/3'))));
                idx_555 = find(~cellfun(@isempty,(strfind(series_names, 'C=3/3'))));
            elseif n_channels == 4
                idx_405 = find(~cellfun(@isempty,(strfind(series_names, 'C=1/4'))));
                idx_488 = find(~cellfun(@isempty,(strfind(series_names, 'C=2/4'))));
                idx_555 = find(~cellfun(@isempty,(strfind(series_names, 'C=3/4'))));
                idx_647 = find(~cellfun(@isempty,(strfind(series_names, 'C=4/4'))));
            end
            
            % move the images for each channel into their own stack
            im_stack_405 = series(idx_405,1);
            im_stack_488 = series(idx_488,1);
            im_stack_555 = series(idx_555,1);
            
            % convert from cell to 3D matrix
            im_stack_405 = cat(3,im_stack_405{:});
            im_stack_488 = cat(3,im_stack_488{:});
            im_stack_555 = cat(3,im_stack_555{:});
            
            % make MIP
            mip_405 = max(im_stack_405, [], 3);
            mip_488 = max(im_stack_488, [], 3);
            mip_555 = max(im_stack_555, [], 3);
            
            if n_channels == 4
                im_stack_647 = series(idx_647,1);
                im_stack_647 = cat(3, im_stack_647{:});
                mip_647 = max(im_stack_647, [], 3);
            end
            
            if show_ims
                figure;
                subplot(1,3,1); imagesc(mip_405); axis off
                subplot(1,3,2); imagesc(mip_488); axis off
                subplot(1,3,3); imagesc(mip_555); axis off
            end
            
            if n_channels == 3 || n_channels == 4
                % store images in MATLAB
                index = length(ims_405) + 1;
                ims_405{index} = mip_405;
                ims_488{index} = mip_488;
                ims_555{index} = mip_555;
                if n_channels == 4
                    ims_647{index} = mip_647;
                end
            end
            
             %extract metadata and create metadata object for save
             metadata_im = createMinimalOMEXMLMetadata(mip_405);
             metadata = data{1,4};
             pixelSizeX = metadata.getPixelsPhysicalSizeX(0).value();
             pixelSizeX = pixelSizeX.doubleValue(); % size of each pixel (X dimension) in microns
             pixelSizeY = metadata.getPixelsPhysicalSizeY(0).value();
             pixelSizeY = pixelSizeY.doubleValue(); % size of each pixel (X dimension) in microns
             area_conv{index} = pixelSizeX * pixelSizeY; % conversion from pixels to microns (microns^2/pixels^2)
             
             pixelSize = ome.units.quantity.Length(java.lang.Double(pixelSizeX), ome.units.UNITS.MICROMETER);
             metadata_im.setPixelsPhysicalSizeX(pixelSize, 0);
             metadata_im.setPixelsPhysicalSizeY(pixelSize, 0);
             
            [px_x, px_y] = size(mip_405);
             length_x = px_x * pixelSizeX;
             length_y = px_y * pixelSizeY;
             
            % write MIP to folder
            cd(im_folder)
            im_name = extractAfter(file_import, append(experiment_name, 'RPMI/'));
            im_name = extractBefore(im_name, '.czi');
            im_name_list{ii} = im_name;
         
                  
            % write MIPs to folder
            bfsave(mip_405, append('max2_', im_name, '_405.tiff'), 'metadata', metadata_im);
            bfsave(mip_488, append('max2_', im_name, '_488.tiff'), 'metadata', metadata_im);
            bfsave(mip_555, append('max2_', im_name, '_555.tiff'), 'metadata', metadata_im);
            %imwrite(mip_405, append('max_', im_name, '_405.tiff'))
            %imwrite(mip_488, append('max_', im_name, '_488.tiff'))
            %imwrite(mip_555, append('max_', im_name, '_555.tiff'))
            if n_channels == 4
                imwrite(mip_647, append('max_', im_name, '_647.tiff'));
            end
            
        end
    end
     waitbar(1, f, 'Done!');
     pause(3)
     close(f)

% for multiple images stored within a single .czi file
elseif ~isempty(path_to_czi)
    
    data = bfopen(path_to_czi{1});
    
    % images within data are stored in this format:
    % each row corresponds to 1 series (containing all channels)
    % within each element of the first column (1,1 2,1 3,1 etc) there is an image file that corresponds to each individual image within the stack
    % images are stored as blue image (z-stack position 1), green image (z-stack position 1), red image (z-stack position 1), blue image (z-stack position 2) etc
    
    % determine number of images within the series
    [n_series,~] = size(data);
    
    for jj = 1:n_series
        
        series = data{jj,1};
        series_names = series(:,2);
        
        % find the images corresponding to each channel
        if n_channels == 3
            idx_405 = find(~cellfun(@isempty,(strfind(series_names, 'C=1/3'))));
            idx_488 = find(~cellfun(@isempty,(strfind(series_names, 'C=2/3'))));
            idx_555 = find(~cellfun(@isempty,(strfind(series_names, 'C=3/3'))));
        elseif n_channels == 4
            idx_405 = find(~cellfun(@isempty,(strfind(series_names, 'C=1/4'))));
            idx_488 = find(~cellfun(@isempty,(strfind(series_names, 'C=2/4'))));
            idx_555 = find(~cellfun(@isempty,(strfind(series_names, 'C=3/4'))));
            idx_647 = find(~cellfun(@isempty,(strfind(series_names, 'C=4/4'))));
        end
        
        % move the images for each channel into their own stack
        im_stack_405 = series(idx_405,1);
        im_stack_488 = series(idx_488,1);
        im_stack_555 = series(idx_555,1);
        
        % convert from cell to 3D matrix
        im_stack_405 = cat(3,im_stack_405{:});
        im_stack_488 = cat(3,im_stack_488{:});
        im_stack_555 = cat(3,im_stack_555{:});
        
        % make MIP
        mip_405 = max(im_stack_405, [], 3);
        mip_488 = max(im_stack_488, [], 3);
        mip_555 = max(im_stack_555, [], 3);
        
        if n_channels == 4
            im_stack_647 = series(ims_647,1);
            im_stack_647 = cat(3,im_stack_647{:});
            mip_647 = max(im_stack_647, [], 3);
        end
        
        % store images in MATLAB
        index = length(ims_405) + 1;
        ims_405{index} = mip_405;
        ims_488{index} = mip_488;
        ims_555{index} = mip_555;
        if n_channels == 4
            ims_647{index} = mip_647;
        end
        
        % write MIP to folder
        experiment_name = extractAfter(path_to_czi{1}, '880/'); % switch from airy to 880
        experiment_name = extractBefore(experiment_name, '.czi');
        
        folder_path = extractBefore(path_to_czi{1}, experiment_name);
        cd(folder_path)
        
        if ~isfolder(experiment_name)
            mkdir(experiment_name)
        end
        
        cd(experiment_name)
        im_name = append('max_', experiment_name, '_n', string(jj));
        
        % write MIPs to folder
        imwrite(mip_405, append(im_name, '_405.tiff'))
        imwrite(mip_488, append(im_name, '_488.tiff'))
        imwrite(mip_555, append(im_name, '_555.tiff'))
        if n_channels == 4
            imwrite(mip_647, append(im_name, '_647.tiff'));
        end
    end
end



%% Background subtraction with mean as background, store files

n_images = length(ims_405);

ims_bsub_405 = {};
ims_bsub_488 = {};
ims_bsub_555 = {};
if n_channels == 4
    ims_bsub_647 = {};
end


for kk = 1:n_images
    
    % 405
    im_405 = ims_405{kk};
    mean_405 = mean(im_405(:));
    ims_bsub_405{kk} = im_405 - mean_405; %background subtraction
    
    % 488
    im_488 = ims_488{kk};
    mean_488 = mean(im_488(:));
    ims_bsub_488{kk} = im_488 - mean_488; %background subtraction
    
    
    % 555
    im_555 = ims_555{kk};
    mean_555 = mean(im_555(:));
    ims_bsub_555{kk} = im_555 - mean_555; % background subtraction
    
    
    if n_channels == 4
        im_647 = ims_647{kk};
        mean_647 = mean(im_647(:));
        ims_bsub_647{kk} = im_647 - mean_647; %background subtraction
    end
    
end


%% Segment cells by phalloidin cell area 
%watershed based on https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/

ims_bsub_647_thresh = {}; %for storing thresholded 647 (phalloidin) images 
n_std_for_thresh_647 = 1; %number of standard deviations to use for threshold for phalloidin stain

for ll = 1:n_images
     %make a cell mask using phalloidin
    im_647 = ims_bsub_647{ll};
    mean_647 = mean(im_647(:));
    mean_647_list{ll} = mean_647;
    std_647 = std(double(im_647(:))); 
    
   %apply threshold to phalloidin stain
    thresh_647 = mean_647 + (std_647*n_std_for_thresh_647); % set adaptive threshold
    im_647_thresh = im_647 > thresh_647; % threshold phalloidin image
   figure; imshow(im_647_thresh);
    
    %morphological operations to improve segmentation 
    im_647_thresh = imfill(im_647_thresh, 'holes');
    SE1 = strel('disk', 1); 
    im_647_thresh = imdilate(im_647_thresh, SE1);
    im_647_thresh = bwareafilt(im_647_thresh, [5000 10000000]);
    SE2 = strel('disk', 2); 
    im_647_thresh = imdilate(im_647_thresh, SE2);
    figure; imshow(im_647_thresh); 
    ims_bsub_647_thresh{ll} = im_647_thresh;
    
    %watershed to separate touching cells
    WD = -bwdist(~im_647_thresh); 
    %figure; imshow(WD, []); 
    WD2 = watershed(WD); 
    %figure; imshow(label2rgb(WD2)); 
    im_647_thresh_WD = im_647_thresh;
    im_647_thresh_WD(WD2 == 0) = 0;
    %figure, imshow(im_647_thresh);
    mask = imextendedmin(WD, 40); 
    %figure; imshowpair(im_647_thresh, mask, 'blend');
    
    WD3 = imimposemin(WD,mask);
    WD4 = watershed(WD3);
    im_647_thresh_WD2 = im_647_thresh;
    im_647_thresh_WD2(WD4 == 0) = 0;
    figure; imshow(im_647_thresh_WD2)

    im_647_thresh_WD2 = imclearborder(im_647_thresh_WD2); %remove objects touching border based on watershed segmentation
    figure; imshow(im_647_thresh_WD2); %check
    
    SE4 = strel('disk', 3); 
    im_647_thresh_final{ll} = imdilate(im_647_thresh_WD2, SE4); %remove lines from watershed segmentation and store final image 
    cell_area_image{ll} = figure; imshow(im_647_thresh_final{ll}); 
    saveas(cell_area_image{ll}, append(im_name_list{ll}, '_cell_area.tif')); %save final thresholded image of cell area
 
    
    %calculate the total cell area per image
    cell_area_pix = cellfun(@nnz, im_647_thresh_final);
    cell_area_pix = num2cell(cell_area_pix); 
    cell_area_micron{ll} = cell_area_pix{ll} * area_conv{ll};


end


%% Segment the nuclei 

ims_bsub_405_thresh = {}; %for storing thresholded DAPI images 
n_std_for_thresh_405 = 1.5; %number of standard deviations to use for threshold for DAPI
im_405_thresh_count = {}; %for storing nuclei images to be counted

for hh = 1:n_images
     %segment the nuclei
    im_405 = ims_bsub_405{hh};
    mean_405 = mean(im_405(:));
    mean_405_list{hh} = mean_405;
    std_405 = std(double(im_405(:))); 
    
   %apply threshold to nuclei
    thresh_405 = mean_405 + (std_405*n_std_for_thresh_405); % set adaptive threshold
    im_405_thresh = im_405 > thresh_405; % threshold nuclei
    im_405_thresh = bwareafilt(im_405_thresh, [5000 10000000]); %remove small objects
    im_405_thresh = imfill(im_405_thresh, 'holes'); % fill any holes in nuclei
   figure; imshow(im_405_thresh);
   
   im_405_thresh = imclearborder(im_405_thresh); 
    figure; imshow(im_405_thresh);
    
    % Make a mask of only nuclei which are found in thresholded (complete)
    % cells
    im_405_thresh_count{hh} = im_405_thresh .* im_647_thresh_final{hh};
    SE3 = strel('disk', 8); 
    im_405_thresh_count{hh} = imdilate(im_405_thresh_count{hh}, SE3); 
    figure; imshow(im_405_thresh_count{hh}); 
    figure; imshow(im_647_thresh_final{hh}); 
    
    %save image of thresholded nuclei
    nuclei_image{hh} = figure; imshow(im_405_thresh_count{hh}); 
    saveas(nuclei_image{hh}, append(im_name_list{hh}, '_nuclei.tif')); % save image of segmented nuclei
   
end




%% Segment based on BD signal and apply cell/nuclei masks
    
ims_bsub_488_thresh = {}; % for storing thresholded BODIPY images
n_std_for_thresh_488 = 2.5; %number of standard deviations to use for threshold for BD
nuclei_count = {}; %place to store nuclei counts


for mm = 1:n_images;
    
    %segment the LDs
    im_488 = ims_bsub_488{mm};
    mean_488 = mean(im_488(:)); 
    mean_488_list{mm} = mean_488; 
    std_488 = std(double(im_488(:))); 
    figure; imshow(im_488, []); 
    
    %apply the threshold to the LDs
    thresh_488 = mean_488 + (std_488*n_std_for_thresh_488); %set adaptive threshold
    im_488_thresh = im_488 > thresh_488; %threshold the LDs
    im_488_thresh = bwareafilt(im_488_thresh, [5, 100000]);
    figure; imshow(im_488_thresh); 
    
    %apply cell mask to thresholded LDs
    im_488_thresh_cell{mm} = im_488_thresh .* im_647_thresh_final{mm};
    figure; imshow(im_488_thresh_cell{mm}); 
    LD_Bodipy{mm} = figure; imshow(im_488_thresh_cell{mm}); 
    saveas(LD_Bodipy{mm}, append(im_name_list{mm}, '_LD_Bodipy.tif')); %save image of segmented LD's
    
    %count the number of nuclei (from im_405_thresh_count (nuclei segmented
    %by cells))
    [labeled_nuclei{mm}, nuclei_count{mm}] = bwlabel(im_405_thresh_count{mm});   
    figure; imshow(labeled_nuclei{mm}); 
    
    %calculate LD area from BODIPY stain
    LD_area_pix_bodipy = cellfun(@nnz, im_488_thresh_cell); 
    LD_area_pix_bodipy = num2cell(LD_area_pix_bodipy);
    LD_area_micron_bodipy{mm} = LD_area_pix_bodipy{mm} * area_conv{mm}; 
    writecell(LD_area_micron_bodipy, 'LD_area_raw_BODIPY.xls');
    
    %normalize LD area to cell area AND number of nuclei
    LD_area_norm2cell_bodipy{mm} = LD_area_micron_bodipy{mm} / cell_area_micron{mm};
    writecell(LD_area_norm2cell_bodipy, 'LD_area_norm2cell_BODIPY.xls');
    LD_area_norm2CandN_bodipy{mm} = LD_area_norm2cell_bodipy{mm} / nuclei_count{mm};
    writecell(LD_area_norm2CandN_bodipy, 'LD_area_norm2CandN_BODIPY.xls');
   
    %normalize LD area to number of nuclei
    LD_area_norm2nuc_output_bodipy{mm} = LD_area_micron_bodipy{mm} / nuclei_count{mm}; 
    writecell(LD_area_norm2nuc_output_bodipy, 'LD_area_norm2nuc_BODIPY.xls'); 
    
    
end




%% Segment the LD's using PLIN2 signal
    
ims_bsub_555_thresh = {}; % for storing thresholded PLIN2 images
n_std_for_thresh_555 = 3.5; %number of standard deviations to use for threshold for PLIN2

for nn = 1:n_images;
    
    %segment the LDs
    im_555 = ims_bsub_555{nn};
    mean_555 = mean(im_555(:)); 
    mean_555_list{nn} = mean_555; 
    std_555 = std(double(im_555(:))); 
    figure; imshow(im_555, []); 
    
    %apply the threshold to the LDs
    thresh_555 = mean_555 + (std_555*n_std_for_thresh_555); %set adaptive threshold
    im_555_thresh = im_555 > thresh_555; %threshold the LDs
    im_555_thresh = bwareafilt(im_555_thresh, [5, 100000]); 
    figure; imshow(im_555_thresh); 
    
    %apply cell mask to thresholded LDs
    im_555_thresh_cell{nn} = im_555_thresh .* im_647_thresh_final{nn};
    figure; imshow(im_555_thresh_cell{nn}); 
    LD_plin2{nn} = figure; imshow(im_555_thresh_cell{nn}); 
    saveas(LD_plin2{nn}, append(im_name_list{nn}, '_LD_plin2.tif')); %save image of segmented LD's
    
    %calculate LD area from PLIN2 stain
    LD_area_pix_plin2 = cellfun(@nnz, im_555_thresh_cell); 
    LD_area_pix_plin2 = num2cell(LD_area_pix_plin2);
    LD_area_micron_plin2{nn} = LD_area_pix_plin2{nn} * area_conv{nn}; 
    writecell(LD_area_micron_plin2, 'LD_area_raw_plin2.xls');
    
    %normalize LD area to cell area AND number fo nuclei
    LD_area_norm2cell_plin2{nn} = LD_area_micron_plin2{nn} / cell_area_micron{nn};
    writecell(LD_area_norm2cell_plin2, 'LD_area_norm2cell_plin2.xls');
    LD_area_norm2CandN_plin2{nn} = LD_area_norm2cell_plin2{nn} / nuclei_count{nn};
    writecell(LD_area_norm2CandN_plin2, 'LD_area_norm2CandN_plin2.xls');
   
    %normalize LD area to number of nuclei
    LD_area_norm2nuc_output_plin2{nn} = LD_area_micron_plin2{nn} / nuclei_count{nn}; 
    writecell(LD_area_norm2nuc_output_plin2, 'LD_area_norm2nuc_plin2.xls');  
    
    
end

close all

