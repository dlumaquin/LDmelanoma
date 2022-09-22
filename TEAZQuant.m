%% QUANTIFICATION OF TEAZ TUMOR IMAGES
% Dianne Lumaquin

clear all
close all

analysis_wt_flag = 1; % run analysis on WT images
analysis_mut_flag = 1; % run analysis on mut images

load_images_flag = 1; % load image .mat files
save_analysis_flag = 1; % save analysis .mat files


showIm = 1;
split_channels = 1; % bioformats plugin: save data for each channel as .mat file
save_server = 1;
chName = {'BF', 'tdT1', 'tdT2', 'tdT3', 'tdT4', 'GFP2'};

% list channels and corresponding #
BF = find(strcmp(chName, 'BF'));
tdT1 = find(strcmp(chName, 'tdT1'));
tdT2 = find(strcmp(chName, 'tdT2'));
tdT3 = find(strcmp(chName, 'tdT3'));
tdT4 = find(strcmp(chName, 'tdT4'));
GFP2 = find(strcmp(chName, 'GFP2'));


folder_wt = %insertpath 
folder_mut = %insertpath
Analysisoutput = %insertpath


%% %% BIOFORMATS .CZI IMPORT - WT

if analysis_wt_flag
    
    fprintf('\nAnalyzing WT images...\n');
    
    filenames_wt = fullfile(folder_wt, '*.czi');
    CZI_files_wt = dir(filenames_wt);
    CZI_file_paths_wt = fullfile(folder_wt, {CZI_files_wt.name});
    
    
if split_channels
        
        % RUN TO SAVE ALL IMAGES/CHANNEL AS 1 .MAT:
        %             for gg = groupNames
        for ii = 1:length(CZI_files_wt);
            reader = bfGetReader(CZI_file_paths_wt{1,ii});
            fprintf('Importing WT image %d/%d\n', ii, length(CZI_files_wt));
            
            BF_wt{1,ii} = bfGetPlane(reader,1);
            tdT1_wt{1,ii} = bfGetPlane(reader,2);
            tdT2_wt{1,ii} = bfGetPlane(reader,3);
            tdT3_wt{1,ii} = bfGetPlane(reader,4);
            tdT4_wt{1,ii} = bfGetPlane(reader,5);
            GFP2_wt{1,ii} = bfGetPlane(reader,6);
        end
        
    end
    
    
    % save to same folder on server where data is from:
    if save_server
        
        path_save_server1 = strcat(fullfile(folder_wt, 'wt_channel1.mat'));
        save([path_save_server1], 'BF_wt')
        path_save_server2 = strcat(fullfile(folder_wt, 'wt_channel2.mat'));
        save([path_save_server2], 'tdT1_wt')
        path_save_server3 = strcat(fullfile(folder_wt, 'wt_channel3.mat'));
        save([path_save_server3], 'tdT2_wt')
        path_save_server4 = strcat(fullfile(folder_wt, 'wt_channel4.mat'));
        save([path_save_server4], 'tdT3_wt')
        path_save_server5 = strcat(fullfile(folder_wt, 'wt_channel5.mat'));
        save([path_save_server5], 'tdT4_wt')
        path_save_server6 = strcat(fullfile(folder_wt, 'wt_channel6.mat'));
        save([path_save_server6], 'GFP2_wt')
    end
    
    % metadata
    reader = bfGetReader(CZI_file_paths_wt{1,1});
    omeMeta = reader.getMetadataStore();
    imageSizeX = omeMeta.getPixelsSizeX(0).getValue();
    imageSizeY = omeMeta.getPixelsSizeY(0).getValue();
    pixelSizeX = omeMeta.getPixelsPhysicalSizeX(0).value();
    pixelSizeX_num = pixelSizeX.doubleValue();
    pixelSizeY = omeMeta.getPixelsPhysicalSizeY(0).value();
    pixelSizeY_num = pixelSizeY.doubleValue();
    pixelSize_um2 = pixelSizeX_num * pixelSizeY_num;
    
    % save metadata
    metadata_wt = [pixelSize_um2 pixelSizeX_num pixelSizeY_num];
    path_save_server_wt = strcat(fullfile(folder_wt, 'metadata_wt.mat'));
    save([path_save_server_wt], 'metadata_wt');
    fprintf('wt metadata .mat saved')
    
    
end

%% BIOFORMATS .CZI IMPORT - MUTANT

if analysis_mut_flag
    
    % mutant
    fprintf('\nAnalyzing mutant images...\n');
    
    filenames_mut = fullfile(folder_mut, '*.czi');
    CZI_files_mut = dir(filenames_mut);
    CZI_file_paths_mut = fullfile(folder_mut, {CZI_files_mut.name});
    
if split_channels
            
            % RUN TO SAVE ALL IMAGES/CHANNEL AS 1 .MAT:
            
            for ii = 1:length(CZI_files_mut);
                reader = bfGetReader(CZI_file_paths_mut{1,ii});
                fprintf('Importing Mut image %d/%d\n', ii, length(CZI_files_mut));
                
                BF_mut{1,ii} = bfGetPlane(reader,1);
                tdT1_mut{1,ii} = bfGetPlane(reader,2);
                tdT2_mut{1,ii} = bfGetPlane(reader,3);
                tdT3_mut{1,ii} = bfGetPlane(reader,4);
                tdT4_mut{1,ii} = bfGetPlane(reader,5);
                GFP2_mut{1,ii} = bfGetPlane(reader,6);
            
            end
            
    end
            
            % save to same folder on server where data is from:
            if save_server
                
                path_save_server1 = strcat(fullfile(folder_mut, 'mut_channel1.mat'));
                save([path_save_server1], 'BF_mut')
                path_save_server2 = strcat(fullfile(folder_mut, 'mut_channel2.mat'));
                save([path_save_server2], 'tdT1_mut')
                path_save_server3 = strcat(fullfile(folder_mut, 'mut_channel3.mat'));
                save([path_save_server3], 'tdT2_mut')
                path_save_server4 = strcat(fullfile(folder_mut, 'mut_channel4.mat'));
                save([path_save_server4], 'tdT3_mut')
                path_save_server5 = strcat(fullfile(folder_mut, 'mut_channel5.mat'));
                save([path_save_server5], 'tdT4_mut')
                path_save_server6 = strcat(fullfile(folder_mut, 'mut_channel6.mat'));
                save([path_save_server6], 'GFP2_mut')
            end
            
     
     
    
    
    % metadata
    reader = bfGetReader(CZI_file_paths_mut{1,1});
    omeMeta = reader.getMetadataStore();
    imageSizeX = omeMeta.getPixelsSizeX(0).getValue();
    imageSizeY = omeMeta.getPixelsSizeY(0).getValue();
    pixelSizeX = omeMeta.getPixelsPhysicalSizeX(0).value();
    pixelSizeX_num = pixelSizeX.doubleValue();
    pixelSizeY = omeMeta.getPixelsPhysicalSizeY(0).value();
    pixelSizeY_num = pixelSizeY.doubleValue();
    pixelSize_um2 = pixelSizeX_num * pixelSizeY_num;
    
    % save metadata
    metadata_mut = [pixelSize_um2 pixelSizeX_num pixelSizeY_num];
    path_save_server_mut = strcat(fullfile(folder_mut, 'metadata_mut.mat'));
    save([path_save_server_mut], 'metadata_mut');
    fprintf('mut metadata .mat saved')




end



%% BACKGROUND CORRECTION VIA GFP: gNT
%goal is to threshold GFP and background correct to capture true tdTomato fluorescence

GFPthresh = 90; %change threshold to catch bg
tdT4thresh = 250; %change threshold to catch tumor cells

for m = 1:length(GFP2_wt)
GFP2_thresh_wt = GFP2_wt{1,m};
GFP2_thresh_wt_bg{1,m} = GFP2_thresh_wt > GFPthresh;

tdT4_thresh_wt = tdT4_wt{1,m};
tdT4_thresh_wt_bg{1,m} = tdT4_thresh_wt > tdT4thresh;
%figure; imshow(tdT4_thresh_wt_bg{1,m})
tdT4_thresh_wt_corr{1,m} = tdT4_thresh_wt_bg{1,m} - GFP2_thresh_wt_bg{1,m};
% figure; imshow(imfuse(BF_wt{1,m},tdT4_thresh_wt_corr{1,m}));

figure ('WindowState','maximized');
subplot(1,2,1), imshow(GFP2_thresh_wt_bg{1,m},[]);
subplot(1,2,2), imshow(tdT4_thresh_wt_corr{1,m},[]);
end

%% BACKGROUND CORRECTION VIA GFP_mutant: gDGAT1

for a = 1:length(GFP2_mut)
GFP2_thresh_mut = GFP2_mut{1,a};
GFP2_thresh_mut_bg{1,a} = GFP2_thresh_mut > GFPthresh;

tdT4_thresh_mut = tdT4_mut{1,a};
tdT4_thresh_mut_bg{1,a} = tdT4_thresh_mut > tdT4thresh;
%figure; imshow(tdT4_thresh_mut_bg{1,a})
tdT4_thresh_mut_corr{1,a} = tdT4_thresh_mut_bg{1,a} - GFP2_thresh_mut_bg{1,a};
%figure; imshow(imfuse(BF_mut{1,a},tdT4_thresh_mut_corr{1,a}));

figure ('WindowState','maximized');
subplot(1,2,1), imshow(GFP2_thresh_mut_bg{1,a},[]);
subplot(1,2,2), imshow(tdT4_thresh_mut_corr{1,a},[]);

end

%% CROP IMAGES: sgNT

crop_wt_flag = 1;
load_wtcrop_flag = 0;

rectpos_xy_wt = NaN(36,4);
cropwindow = [0,0,1200,1000]; % change dimensions of cropped window as needed
vehfile = fullfile(folder_wt,'croppositions_sgNT.csv');

if crop_wt_flag
%crop tdTomato image
for cc = 1:length(tdT2_wt)% modify
I = imfuse(BF_wt{1,cc},tdT4_thresh_wt_corr{1,cc});
figure; imshow(I)
rect = drawrectangle('Position',cropwindow,'Color','r');
pause; %make sure square is around area of interest and then press return key
currkey=get(gcf,'CurrentKey'); 
        if currkey=='return' 
            currkey==1
        else
            currkey==0
        end
rectpos = get(rect,'Position');
rectpos_xy_wt(cc,1:4) = rectpos;
crop = imcrop(tdT4_thresh_wt_corr{1,cc},rectpos_xy_wt(cc,1:4));
%figure;imshow(crop)
tdT4_thresh_wt_crop{1,cc} = crop;   
end
close all
writematrix(rectpos_xy_wt,vehfile);
end

if load_wtcrop_flag
for cc = 1:length(tdT2_wt)% modify
    I = tdT4_thresh_wt_corr{1,cc};
    rectpos_xy_wt = readmatrix(vehfile);
    crop = imcrop(I,rectpos_xy_wt(cc,1:4));
    %figure;imshow(crop)
    tdT4_thresh_wt_crop{1,cc} = crop;
end
end

%% CROP IMAGES: sgDGAT1

crop_mut_flag = 1;
load_mutcrop_flag = 0;

rectpos_xy_mut = NaN(36,4);
vehfile = fullfile(folder_mut,'croppositions_sgDGAT.csv');

if crop_mut_flag
%crop tdTomato image
for cc = 1:length(tdT2_mut)% modify
I = imfuse(BF_mut{1,cc},tdT4_thresh_mut_corr{1,cc});
figure; imshow(I)
rect = drawrectangle('Position',cropwindow,'Color','r');
pause; %make sure circle is around area of interest and then press return key
currkey=get(gcf,'CurrentKey');  
        if currkey=='return' 
            currkey==1
        else
            currkey==0
        end
rectpos = get(rect,'Position');
rectpos_xy_mut(cc,1:4) = rectpos;
crop = imcrop(tdT4_thresh_mut_corr{1,cc},rectpos_xy_mut(cc,1:4));
%figure;imshow(crop)
tdT4_thresh_mut_crop{1,cc} = crop;   
end
close all
writematrix(rectpos_xy_mut,vehfile);
end

if load_mutcrop_flag
for cc = 1:length(tdT2_mut)% modify
    I = tdT4_thresh_mut_corr{1,cc};
    rectpos_xy_mut = readmatrix(vehfile);
    crop = imcrop(I,rectpos_xy_mut(cc,1:4));
    %figure;imshow(crop)
    tdT4_thresh_mut_crop{1,cc} = crop;
end
end


%% FINAL SEGMENTATION sgNT

BFthresh = 390;
%BF thresholding
for n = 1:length(tdT2_wt)
BF_thresh_wt{1,n} = BF_wt{1,n} < BFthresh;
BF_corr_wt{1,n} = BF_thresh_wt{1,n} - GFP2_thresh_wt_bg{1,n};

%crop images
BF_corr_wt{1,n} = imcrop(BF_corr_wt{1,n},rectpos_xy_wt(n,1:4));
BF_wt_crop{1,n} = imcrop(BF_wt{1,n},rectpos_xy_wt(n,1:4));

figure ('WindowState','maximized');
subplot(1,2,1), imshow(imfuse(BF_wt_crop{1,n},BF_corr_wt{1,n}));
subplot(1,2,2), imshow(imfuse((tdT4_thresh_wt_crop{1,n}>0), (BF_corr_wt{1,n}>0)));
end

% Final overlay and save
for n = 1:length(tdT2_wt)
BF_tdT_wt{1,n} = BF_corr_wt{1,n} + tdT4_thresh_wt_crop{1,n};
BF_tdT_thresh_wt{1,n} = BF_tdT_wt{1,n} > 0;
%figure; imagesc(BF_tdT_thresh_wt{1,n});

figure; image_wt = imshow(imfuse(BF_wt_crop{1,n}, BF_tdT_thresh_wt{1,n}));
baseFileName_wt = sprintf('wt_corr%d.tif',n);
fullFileName_wt = fullfile(folder_wt, baseFileName_wt);
% saveas(image_wt,fullFileName_wt,'tif');
end

%% FINAL SEGMENTATION: sgDGAT1

for nn = 1:length(tdT2_mut)
BF_thresh_mut{1,nn} = BF_mut{1,nn} < BFthresh;
BF_corr_mut{1,nn} = BF_thresh_mut{1,nn} - GFP2_thresh_mut_bg{1,nn};

%crop images
BF_corr_mut{1,nn} = imcrop(BF_corr_mut{1,nn},rectpos_xy_mut(nn,1:4));
BF_mut_crop{1,nn} = imcrop(BF_mut{1,nn},rectpos_xy_mut(nn,1:4));

figure ('WindowState','maximized');
subplot(1,2,1), imshow(imfuse(BF_mut_crop{1,nn},BF_corr_mut{1,nn}));
subplot(1,2,2), imshow(imfuse((tdT4_thresh_mut_crop{1,nn}>0), (BF_corr_mut{1,nn}>0)));
end


for nn = 1:length(tdT2_mut)
BF_tdT_mut{1,nn} = BF_corr_mut{1,nn} + tdT4_thresh_mut_crop{1,nn};
BF_tdT_thresh_mut{1,nn} = BF_tdT_mut{1,nn} > 0;
%figure; imagesc(BF_tdT_thresh_mut{1,nn});

figure; image_mut = imshow(imfuse(BF_mut_crop{1,nn}, BF_tdT_thresh_mut{1,nn}));
baseFileName_mut = sprintf('mut_corr%d.tif',nn);
fullFileName_mut = fullfile(folder_mut, baseFileName_mut);
saveas(image_mut,fullFileName_mut,'tif');
end


%% Quantification

pxCount_tdT2 = NaN(36,2);
for aa = 1:length(tdT2_wt)
    pxCount_tdT2(aa,1) = sum(tdT4_thresh_wt_corr{1,aa}(:) == 1);
end

for bb = 1:length(tdT2_mut)
    pxCount_tdT2(bb,2) = sum(tdT4_thresh_mut_corr{1,bb}(:) == 1);
end

px_tdT_table = table(pxCount_tdT2(:,1), pxCount_tdT2(:,2),'VariableNames', {'gNT', 'gDGAT1a'});

%make sure analysis file output is changed

file_date = sprintf('TEAZpixelcount_%s.csv', datestr(now,'mm-dd-yyyy'));
Filename = fullfile(Analysisoutput, file_date);
writetable(px_tdT_table,Filename);

%convert into area
area_tdT2 = NaN(36,2);
for ff = 1:length(tdT2_wt)
    area_tdT2(ff,1) = pxCount_tdT2(ff,1) .* (metadata_wt(1,1)/1e6);
end

for gg = 1:length(tdT2_mut)
    area_tdT2(gg,2) = pxCount_tdT2(gg,2) .* (metadata_mut(1,1)/1e6);
end

area_tdT_table = table(area_tdT2(:,1), area_tdT2(:,2),'VariableNames', {'gNT mm2', 'gDGAT1a mm2'});

area_file_date = sprintf('TEAZarea_%s.csv', datestr(now,'mm-dd-yyyy'));
Areafilename = fullfile(Analysisoutput, area_file_date);
writetable(area_tdT_table,Areafilename);



fprintf('Analysis Complete')