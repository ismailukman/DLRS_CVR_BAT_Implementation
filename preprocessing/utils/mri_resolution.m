function resolution = mri_resolution(M)
% mri_resolution  Voxel size in mm from an affine matrix.
%
%   resolution = mri_resolution(M)
%
%   M   4x4 (or 3x3) affine. Only the 3x3 rotation/scale block is used.
%
%   Returns the length of each column of that block, which is the voxel
%   dimension along x, y and z regardless of any rotation in the affine.
    M = M(1:3, 1:3);
    resolution = sqrt(sum(M.^2, 1));
end