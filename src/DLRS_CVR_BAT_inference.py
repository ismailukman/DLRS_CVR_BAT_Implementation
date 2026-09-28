# %%
import os
import fnmatch
import numpy as np
import torch
from torch.utils import data
import nibabel as nib
from DLRS_CVR_BAT_model import UNet_dual


class CVRDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_root):
        'initialization'
        self.input_IDs = []
        self._scan_dir(data_root)

    def _scan_dir(self, search_dir):
        'Recursively scan for DLRS_input folders (supports group subdirs)'
        for entry in os.listdir(search_dir):
            entry_path = os.path.join(search_dir, entry)
            if not os.path.isdir(entry_path) or entry == "output":
                continue
            dlrs_input_path = os.path.join(entry_path, "DLRS_input")
            if os.path.exists(dlrs_input_path):
                # Found a subject with DLRS_input  - collect layer files
                files_in_dlrs_input = [
                    os.path.join(dlrs_input_path, file) for file in os.listdir(dlrs_input_path)
                    if fnmatch.fnmatch(file, 'DLRS_input_layer_[0-9][0-9][0-9].nii')
                ]
                self.input_IDs.extend(files_in_dlrs_input)
            else:
                # May be a group folder (HC_subj, SCI_subj)  - recurse one level
                self._scan_dir(entry_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load input
        filename, extension = os.path.splitext(self.input_IDs[index])
        img = nib.load(self.input_IDs[index])

        image, img_affine = img.get_fdata(), img.affine
        image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image).type(torch.FloatTensor)
        img.uncache()

        return [filename, image, img_affine]


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def main():

    # Resolve paths relative to the project root (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    wkdir = os.path.join(project_root, 'data')
    model_path = os.path.join(project_root, 'model', 'pretrained_DLRS_CVR_BAT_model.pt')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    inference_set = CVRDataset(wkdir)
    data_loader = data.DataLoader(inference_set, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    model = UNet_dual(in_channels=136, n_classes=1, depth=5, batch_norm=True, padding=True, up_mode='upconv')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for X_ID, X, X_affine in data_loader:
            X_dir = X_ID[0]
            X = X.to(device)
            prediction = model(X)  # [N, H, W]

            cvr = np.squeeze(prediction[0, 0, 2:-3, 1:-2].cpu().clone().numpy())
            X_affine = np.squeeze(X_affine.cpu().clone().numpy())
            img_output = nib.Nifti1Image(cvr, X_affine)
            nib.save(img_output, '_'.join([X_dir, 'CVR.nii']))

            bat = np.squeeze(prediction[0, 1, 2:-3, 1:-2].cpu().clone().numpy())
            bat_output = nib.Nifti1Image(bat, X_affine)
            nib.save(bat_output, '_'.join([X_dir, 'BAT.nii']))

    output_dir = "/".join([wkdir, "output"])
    create_folder(output_dir)

    def assemble_3d_outputs(search_dir, output_base):
        """Scan for DLRS_input folders and assemble 3D NIfTI outputs."""
        for subdir in sorted(os.listdir(search_dir)):
            subdir_path = os.path.join(search_dir, subdir)
            if not os.path.isdir(subdir_path) or subdir == "output":
                continue
            dlrs_input_path = os.path.join(subdir_path, "DLRS_input")
            if os.path.exists(dlrs_input_path):
                # Compute relative path from wkdir for output structure
                rel_path = os.path.relpath(subdir_path, wkdir)
                out_path = os.path.join(output_base, rel_path)
                create_folder(out_path)

                cvr_files = sorted([
                    os.path.join(dlrs_input_path, f) for f in os.listdir(dlrs_input_path)
                    if fnmatch.fnmatch(f, 'DLRS_input_layer_[0-9][0-9][0-9]_CVR.nii')
                ])
                bat_files = sorted([
                    os.path.join(dlrs_input_path, f) for f in os.listdir(dlrs_input_path)
                    if fnmatch.fnmatch(f, 'DLRS_input_layer_[0-9][0-9][0-9]_BAT.nii')
                ])

                if cvr_files:
                    cvr_3D = np.dstack([nib.load(f).get_fdata() for f in cvr_files])
                    nib.save(nib.Nifti1Image(cvr_3D, X_affine),
                             os.path.join(out_path, "DLRS_CVR.nii"))
                if bat_files:
                    bat_3D = np.dstack([nib.load(f).get_fdata() for f in bat_files])
                    nib.save(nib.Nifti1Image(bat_3D, X_affine),
                             os.path.join(out_path, "DLRS_BAT.nii"))

                print(f"  Assembled 3D outputs: {rel_path}")
            else:
                # Recurse into group folders (HC_subj, SCI_subj)
                assemble_3d_outputs(subdir_path, output_base)

    assemble_3d_outputs(wkdir, output_dir)


if __name__ == "__main__":
    main()
