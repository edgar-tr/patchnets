# PatchNets

This is the official repository for the project "PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations". For details, we refer to [our project page](http://gvv.mpi-inf.mpg.de/projects/PatchNets/), which also includes supplemental videos.

<img src="https://raw.githubusercontent.com/edgar-tr/patchnets/master/misc/teaser.png" width="383" height="371">

This code requires a functioning installation of [DeepSDF](https://github.com/facebookresearch/DeepSDF), which can then be modified using the provided files.

## (Optional) Making ShapeNet V1 Watertight

If you want to use ShapeNet, please follow these steps:

1. Download [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks)
2. On Linux, follow the installation steps from there:
```
conda env create -f environment.yaml
conda activate mesh_funcspace
python setup.py build_ext --inplace
```
3. Install the four external dependencies from `external/mesh-fusion`:
    * for `libfusioncpu` and `libfusiongpu`, run `cmake` and then `setup.py`
    * for `libmcubes` and `librender`, run `setup.py`
4. Replace the original OccNet files with the included slightly modified versions. This mostly switches to using `.obj` instead of `.off`
5. Prepare the original Shapenet meshes by copying all objs as follows: from `02858304/1b2e790b7c57fc5d2a08194fd3f4120d/model.obj` to `02858304/1b2e790b7c57fc5d2a08194fd3f4120d.obj`
6. Use `generate_watertight_meshes_and_sample_points()` from `useful_scripts.py`. Needs to be run twice, see comment at `generate_command`.
7. On a Linux machine with display, `activate mesh_funcspace`
8. Run the generated `command.sh`. Note: this preprocessing crashes frequently because some meshes cause issues. They need to be deleted.

## Preprocessing

During preprocessing, we generate SDF samples from obj files.

The C++ files in `src/` are modified versions of the corresponding DeepSDF files. Please follow the instruction on the [DeepSDF github repo](https://github.com/facebookresearch/DeepSDF) to compile these. Then run `preprocess_data.py`. There is a special flag in `preprocess_data.py` for easier handling of ShapeNet. There is also an example command for preprocessing ShapeNet in the code comments. If you want to use depth completion, add the `--randomdepth` and `--depth` flags to the call to `preprocess_data.py`.

## Training

The files in `code/` largely follow DeepSDF and replace the corresponding files in your DeepSDF installation. Note that some legacy functions from these files might not be compatible with PatchNets.
* Some settings files are available in `code/specs/`. The training/test splits can be found in `code/examples/splits/`. The `DataSource` and, if used, the `patch_network_pretrained_path` and `pretrained_depth_encoder_weights` need to be adapted.
* Set a folder that collects all experiments in `code/localization/SystemSpecific.py`.
* The code uses `code/specs.json` as the settings file. Replace this file with the desired settings file.
* The code creates a results folder, which also includes a backup. This is necessary to later run the evaluation script.
* Throughout the code, `metadata` refers to patch extrinsics.
* `mixture_latent_mode` can be set to `all_explicit` for normal PatchNets mode or to `all_implicit` for use with object latents.
  * Some weights automatically change in `deep_sdf_decoder.py` depending on whether `all_explicit` or `all_implicit` is used.
* For all_implicit/object latents, please set `sdf_filename` under `use_precomputed_bias_init` in `deep_sdf_decoder.py` to an `.npz` file that was obtained via Preprocessing and for which `initialize_mixture_latent_vector()` from `train_deep_sdf.py` has been run (e.g. by including it in the training set and training a normal PatchNet). `MixtureCodeLength` is the object latent size and `PatchCodeLength` is the size of each of the regressed patch codes.
* For all_explicit/normal PatchNets, `MixtureCodeLength` needs to be compatible with `PatchCodeLength`. Set `MixtureCodeLength = (PatchCodeLength + 7) x num_patches`. The 7 comes from position (3) + rotation (3) + scale (1). Always use 7, regardless of whether scale and/or rotation are used. Consider keeping the patch extrinsics fixed at their initial values instead of optimizing them with the extrinsics loss, see the second stage of `StagedTraining`.
* When using staged training, `NumEpochs` and the total Lengths of each `Staged` schedule should be equal. Also note that both `Staged` schedules should have the exact same `Lengths` list.

## Evaluation

1. Fit PatchNets to test data: Use `train_deep_sdf.py` to run the trained network on the test data. Getting the patch parameters for a test set is almost the same workflow as training a network, except that the network weights are initialized and then kept fixed and a few other settings are changed. Please see included test `specs.json` for examples. In all cases, set `test_time = True`, `train_patch_network = False`, `train_object_to_patch = False`. Set `patch_network_pretrained_path` in the test `specs.json` to the results folder of the trained network. Make sure that `ScenesPerBatch` is a multiple of the test set size. Adjust the learning rate schedules according to the test `specs.json` examples included.
2. Get quantitative evaluation: Use `evaluate_patch_network_metrics()` from `useful_scripts.py` with the test results folder. Needs to be run twice, see comment at `generate_meshes`. Running this script requires an installation of [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks), see comments in `evaluate_patch_network_metrics()`. It also requires the `obj` files of the dataset that were used for Preprocessing.

## Applications, Experiments, and Mesh Extraction

`useful_scripts.py` contains code for the object latent applications from Sec. 4.3: latent interpolation, the generative model and depth completion. The depth completion code contains a mode for quantitative evaluation. `useful_scripts.py` also contains code to extract meshes.

`code/deep_sdf/data.py` contains the code snippet used for the synthetic noise experiments in Sec. 7 of the supplementary material.

## Additional Functionality

The code contains additional functionalities that are not part of the publication. They might work but have not been thoroughly tested and can be removed.
* wrappers to allow for easy interaction with a trained network (do not remove, required to run evaluation)
  * `_setup_torch_network()` in `useful_scripts.py`
* a patch encoder
  * Instead of autodecoding a patch latent code, it is regressed from SDF point samples that lie inside the patch.
  * `Encoder` in `specs.json`. Check that this works as intended, later changes to the code might have broken something.
* a depth encoder 
  * A depth encoder maps from one depth map to all patch parameters.
  * `use_depth_encoder` in `specs.json`. Check that this works as intended, later changes to the code might have broken something.
* a tiny PatchNet version
  * The latent code is reshaped and used as network weights, i.e. there are no shared weights between different patches.
  * `dims` in `specs.json` should be set to something small like [ 8, 8, 8, 8, 8, 8, 8 ]
  * `use_tiny_patchnet` in `specs.json`
  * Requires to set `PatchLatentCode` correctly, the desired value is printed by `_initialize_tiny_patchnet()` in `deep_sdf_decoder.py`.
* a hierarchical representation
  * Represents/encodes a shape using large patches for simple regions and smaller patches for complex regions of the geometry.
  * `hierarchical_representation()` in `useful_scripts.py`. Never tested. Later changes to the network code might also have broken something.
* simplified curriculum weighting from [Curriculum DeepSDF](https://github.com/haidongz-usc/Curriculum-DeepSDF)
  * `use_curriculum_weighting` in `specs.json`. Additional parameters are in `train_deep_sdf.py`. This is our own implementation, not based on their repo, so mistakes are ours.
* positional encoding from [NeRF](https://github.com/bmild/nerf)
  * `positional_encoding` in `specs.json`. Additional parameters are in `train_deep_sdf.py`. This is our own implementation, not based on their repo, so mistakes are ours.
* a [Neural ODE](https://github.com/rtqichen/torchdiffeq) deformation model for patches
  * Instead of a simple MLP regressing the SDF value, a velocity field first deforms the patch region and then the z-value of the final xyz position is returned as the SDF value. Thus the field flattens the surface to lie in the z=0 plane. Very slow due to Neural ODE. Might be useful to get UV maps/a direct surface parametrization.
  * `use_ode` and `time_dependent_ode` in `specs.json`. Additional parameters are in `train_deep_sdf.py`.
* a mixed representation that has explicit patch latent codes and only regresses patch extrinsics from an object latent code
  * Set `mixture_latent_mode` in `specs.json` to `patch_explicit_meta_implicit`. `posrot_latent_size` is the size of the object latent code in this case. `mixture_to_patch_parameters` is the network that regresses the patch extrinsics. Check that this works as intended, later changes to the code might have broken something.
	
## Citation

This code builds on DeepSDF. Please consider citing DeepSDF and PatchNets if you use this code.
```
@article{Tretschk2020PatchNets, 
    author = {Tretschk, Edgar and Tewari, Ayush and Golyanik, Vladislav and Zollh\"{o}fer, Michael and Stoll, Carsten and Theobalt, Christian}, 
    title = "{PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations}", 
    journal = {European Conference on Computer Vision (ECCV)}, 
    year = "2020" 
} 
@InProceedings{Park_2019_CVPR,
    author = {Park, Jeong Joon and Florence, Peter and Straub, Julian and Newcombe, Richard and Lovegrove, Steven},
    title = {DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```

## License

Please note that this code is released under an MIT licence, see `LICENCE`. We have included and modified third-party components, which have their own licenses. We thank all of the respective authors for releasing their code, especially the team behind DeepSDF!