function img3Dto2D(CVR_g_4D, prefix_mri, d, ud, lr)
% img3Dto2D  Write a 4D volume out as per-slice 2D NIfTI files for the network.
%
%   img3Dto2D(CVR_g_4D, prefix_mri, d, ud, lr)
%
%   CVR_g_4D    4D array [x y z channel]. Channels are the 133 ROI correlation
%               maps followed by the CVR beta0, beta1 and BAT prior.
%   prefix_mri  output path prefix; each slice is written as <prefix><NNN>.nii
%   d           slice direction: 1 = axial (z, slices 1:91),
%               2 = coronal (y, slices 15:95), 3 = sagittal (x, slices 15:74)
%   ud, lr      flip flags used to write mirrored copies for augmentation.
%               ud = 1 flips up/down, lr = 1 flips left/right. The first digit
%               of the output index encodes the variant: 0 = none, 1 = ud,
%               2 = lr.
%
%   Each slice is zero-padded from the native 91 x 109 grid to the 96 x 112
%   the network expects (2 rows and 1 column before, 3 rows and 2 columns
%   after). Existing files at the target name are deleted before writing.

mni_resolution = [2, 2, 2];
mni_type = 16;

if d == 1 %z direction
    
    z_ini = 1;
    z_end = 91;
    
    for kk = z_ini:z_end
        
        bold_img = squeeze(CVR_g_4D(:,:, kk, :));
        
        if ud == 0 && lr == 0
            
            bold_img_f = padarray(bold_img,[2 1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 2],0,'post');
            
            if kk<10
                delete([prefix_mri, '00', num2str(kk), '.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '00', num2str(kk), '.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '0', num2str(kk), '.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '0', num2str(kk), '.nii'], mni_resolution, mni_type);
            end
        end
        
        if ud == 1 && lr == 0
            bold_img = flipud(bold_img);
            bold_img_f = padarray(bold_img,[2 1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 2],0,'post');
            if kk<10
                delete([prefix_mri, '10', num2str(kk), '.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '10', num2str(kk), '.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '1', num2str(kk), '.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '1', num2str(kk), '.nii'], mni_resolution, mni_type);
            end
        end
        
        if ud == 0 && lr == 1
            bold_img = fliplr(bold_img);
            bold_img_f = padarray(bold_img,[2 1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 2],0,'post');
            if kk<10
                delete([prefix_mri, '20', num2str(kk), '.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '20', num2str(kk), '.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '2', num2str(kk), '.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '2', num2str(kk), '.nii'], mni_resolution, mni_type);
            end
        end
        
    end
end

if d == 2 %y direction (coronal write)
    
    y_ini = 15;
    y_end = 95;
    
    for kk = y_ini:y_end
        
        bold_img = squeeze(CVR_g_4D(:, kk, :, :));
        
        if ud == 0 && lr == 0
            
            bold_img_f = padarray(bold_img,[2 9+1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 9+2],0,'post');
            
            if kk<10
                delete([prefix_mri, '00', num2str(kk), 'y.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '00', num2str(kk), 'y.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '0', num2str(kk), 'y.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '0', num2str(kk), 'y.nii'], mni_resolution, mni_type);
            end
        end
        
        if ud == 1 && lr == 0
            bold_img = flipud(bold_img);
            bold_img_f = padarray(bold_img,[2 9+1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 9+2],0,'post');
            if kk<10
                delete([prefix_mri, '10', num2str(kk), 'y.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '10', num2str(kk), 'y.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '1', num2str(kk), 'y.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '1', num2str(kk), 'y.nii'], mni_resolution, mni_type);
            end
        end
        
        if ud == 0 && lr == 1
            bold_img = fliplr(bold_img);
            bold_img_f = padarray(bold_img,[2 9+1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 9+2],0,'post');
            if kk<10
                delete([prefix_mri, '20', num2str(kk), 'y.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '20', num2str(kk), 'y.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '2', num2str(kk), 'y.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '2', num2str(kk), 'y.nii'], mni_resolution, mni_type);
            end
        end
        
    end
end


if d == 3 %x direction (sagittal write)
    
    x_ini = 15;
    x_end = 74;
    
    for kk = x_ini:x_end
        
        bold_img = permute(squeeze(CVR_g_4D(kk, :, :, :)), [2 1 3]);
        
        if ud == 0 && lr == 0
            
            bold_img_f = padarray(bold_img,[2 1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 2],0,'post');
            
            if kk<10
                delete([prefix_mri, '00', num2str(kk), 'z.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '00', num2str(kk), 'z.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '0', num2str(kk), 'z.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '0', num2str(kk), 'z.nii'], mni_resolution, mni_type);
            end
        end
        
        if ud == 1 && lr == 0
            bold_img = flipud(bold_img);
            bold_img_f = padarray(bold_img,[2 1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 2],0,'post');
            if kk<10
                delete([prefix_mri, '10', num2str(kk), 'z.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '10', num2str(kk), 'z.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '1', num2str(kk), 'z.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '1', num2str(kk), 'z.nii'], mni_resolution, mni_type);
            end
        end
        
        if ud == 0 && lr == 1
            bold_img = fliplr(bold_img);
            bold_img_f = padarray(bold_img,[2 1],0,'pre');
            bold_img_f = padarray(bold_img_f,[3 2],0,'post');
            if kk<10
                delete([prefix_mri, '20', num2str(kk), 'z.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '20', num2str(kk), 'z.nii'], mni_resolution, mni_type);
            else
                delete([prefix_mri, '2', num2str(kk), 'z.nii']);
                write_hdrimg(bold_img_f, [prefix_mri, '2', num2str(kk), 'z.nii'], mni_resolution, mni_type);
            end
        end
        
    end
end

end
