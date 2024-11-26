import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

test_load = nib.load('./NeuroIMAGE/sub-0027000/ses-1/anat/sub-0027000_ses-1_run-1_T1w.nii').get_fdata()
test = test_load[:,:,59]
plt.imshow(test)
plt.show()

for i in range(5):
    plt.subplot(5, 5,i + 1)
    plt.imshow(test[:,:,59 + i])
    plt.gcf().set_size_inches(10, 10)
plt.show()
