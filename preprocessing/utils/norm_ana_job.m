function norm_ana_job(target, deform)
% norm_ana_job  Apply an existing SPM deformation field to one image.
%
%   norm_ana_job(target, deform)
%
%   target  image to resample
%   deform  deformation field (y_*.nii) from a prior segmentation
%
%   Writes the normalised image with a 'w' prefix on a 2 mm isotropic grid
%   with bounding box [-90 -126 -72; 90 90 108], using 4th-degree B-spline
%   interpolation.
matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deform};

matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {target};
matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72
                                                          90 90 108];
matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
spm_jobman('initcfg')
spm_jobman('run',matlabbatch)
return