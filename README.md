# Motion Artifacts Reduction in Brain MRI by means of a Deep Residual Network with Densely Connected Multi-resolution Blocks (DRN-DCMB)

Liu J, Kocak M, Supanich M, Deng J. Motion artifacts reduction in brain MRI by means of a deep residual network with densely connected multi-resolution blocks (DRN-DCMB). Magn Reson Imaging. 2020;71. doi:10.1016/j.mri.2020.05.002

Abstract:

Objective: Magnetic resonance imaging (MRI) acquisition is inherently sensitive to motion, and motion artifact reduction is essential for improving image quality in MRI. Methods: We developed a deep residual network with densely connected multi-resolution blocks (DRN-DCMB) model to reduce the motion artifacts in T1 weighted (T1W) spin echo images acquired on different imaging planes before and after contrast injection. The DRN-DCMB network consisted of multiple multi-resolution blocks connected with dense connections in a feedforward manner. A single residual unit was used to connect the input and output of the entire network with one shortcut connection to predict a residual image (i.e. artifact image). The model was trained with five motion-free T1W image stacks (pre-contrast axial and sagittal, and post-contrast axial, coronal, and sagittal images) with simulated motion artifacts. Results: In other 86 testing image stacks with simulated artifacts, our DRN-DCMB model outperformed other state-of-the-art deep learning models with significantly higher structural similarity index (SSIM) and improvement in signal-to-noise ratio (ISNR). The DRN-DCMB model was also applied to 121 testing image stacks appeared with various degrees of real motion artifacts. The acquired images and processed images by the DRN-DCMB model were randomly mixed, and image quality was blindly evaluated by a neuroradiologist. The DRN-DCMB model significantly improved the overall image quality, reduced the severity of the motion artifacts, and improved the image sharpness, while kept the image contrast. Conclusion:  Our DRN-DCMB model provided an effective method for reducing motion artifacts and improving the overall clinical image quality of brain MRI.

Keywords: MRI, motion artifact, deep leaning, multi-resolution block, dense connection, residual learning



