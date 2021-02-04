import numpy as np
import shutil
import os
import pathlib
import sys
sys.stdout.flush()
from multiprocessing import Pool

def _write_mesh(vertex_pos, faces, output_file, offset=[0., 0., 0.], color=None, normal=None):
    with open(output_file, "w") as mesh_file:
        for i, (x, y, z) in enumerate(vertex_pos + offset):
            s = ""
            if normal is not None:
                nx, ny, nz = normal[i,:]
                s += "vn " + str(nx) + " " + str(ny) + " " + str(nz) + "\n"
            s += "v " + str(x) + " " + str(y) + " " + str(z)
            if color is not None:
                r, g, b = color[i,:]
                s += " " + str(r) + " " + str(g) + " " + str(b)
            s += "\n"
            mesh_file.write(s)
        if faces is None:
            return
        elif type(faces) is str:
            mesh_file.write(faces)
        else:
            for x, y, z in faces:
                if normal is None:
                    mesh_file.write("f " + str(x) + " " + str(y) + " " + str(z) + "\n")
                else:
                    mesh_file.write("f " + str(x) + "//" + str(x) + " " + str(y) + "//" + str(y) + " " + str(z) + "//" + str(z) + "\n")


def visualize_mixture(results_folder, grid_resolution=None, mesh_files=None, data_source=None, break_if_latent_does_not_exist=False, output_name=None, checkpoints=None):
    import torch

    if checkpoints is None:
        checkpoints = "latest"
    checkpoints = [str(checkpoint) for checkpoint in (checkpoints if type(checkpoints) == list else [checkpoints])]
    if output_name is None:
        output_name = "parts_mixture"
        output_name += "/"
    if grid_resolution is None:
        grid_resolution = 128
    if mesh_files is None:
        mesh_files = None
    if data_source is None:
        data_source = None

    output_folder = results_folder + "output/parts/" + output_name

    create_folder(output_folder)

    network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs = _setup_torch_network(results_folder)
    network.use_depth_encoder = False

    with open(results_folder + "backup/special_files/npyfilenames.txt", "r") as train_split:
        train_split = train_split.readlines()
        train_split = [line[:-5] for line in train_split] # remove ".npz\n"

    if mesh_files is None:
        mesh_files = train_split
    elif type(mesh_files) != list:
        import json
        if mesh_files[-5:] == ".json":
            with open(mesh_files, "r") as mesh_files:
                mesh_files_from_json = list(json.load(mesh_files).items())
            mesh_files = []
            for super_folder, entries in mesh_files_from_json:
                for subfolder, subentries in entries.items():
                    mesh_files += [super_folder + "/" + subfolder + "/" + mesh_file for mesh_file in subentries]
        else:
            with open(mesh_files, "r") as mesh_files:
                mesh_files = mesh_files.readlines()[:-1]
        with open(results_folder + "backup/specs.json", "r") as specs:
            specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
            specs = json.loads(specs)

    if data_source is None:
        data_source = specs["DataSource"] + "SdfSamples/" 
             
    with open(output_folder + "MeshFiles", "w") as output_mesh_files:
        for mesh_file in mesh_files:
            for checkpoint in checkpoints:
                output_mesh_files.write(checkpoint + "/" + mesh_file + ".obj\n")
                
    for checkpoint in checkpoints:
        checkpoint_output_folder = output_folder + checkpoint + "/"
        create_folder(checkpoint_output_folder)
        
        with open(checkpoint_output_folder + "MeshFiles", "w") as output_mesh_files:
            for mesh_file in mesh_files:
                output_mesh_files.write(mesh_file + ".obj\n")
                
        load_weights_into_network(network, checkpoint=checkpoint)
        train_latent_codes = torch.load(results_folder + "LatentCodes/" + checkpoint + ".pth")
        for i, mesh in enumerate(mesh_files):
            if mesh in train_split:
                train_id = train_split.index(mesh)
                latent_code = train_latent_codes["latent_codes"][train_id,0,:].unsqueeze(0).clone().cuda()
            else:
                sdf_filename = data_source + mesh + ".npz"
                sdf = np.load(sdf_filename)
                sdf = (torch.from_numpy(sdf["pos"]), torch.from_numpy(sdf["neg"]))
                _, latent_code = sdf_to_latent(sdf, full_loss=True, latent_init=latent_code, num_samples=3072, num_iterations=1000, sdf_filename=sdf_filename, lr=0.001, adjust_lr_every=200, decrease_lr_by=2)
                                
            vertices, faces = latent_to_mesh(latent_code, grid_resolution=grid_resolution)
            mesh_folder = os.path.split(checkpoint_output_folder + mesh)[0]
            create_folder(mesh_folder)
            _write_mesh(vertices, faces+1, checkpoint_output_folder + mesh + ".obj")
            torch.save( {"epoch": None, "latent_codes": latent_code.reshape(1,1,-1)}, checkpoint_output_folder + mesh + "_latent.pth")


def _setup_torch_network(results_folder, checkpoint="latest"):
    import torch

    extra_sys_folder = results_folder + "backup/"

    import sys
    sys.path.append(extra_sys_folder)
    from networks.deep_sdf_decoder import Decoder as Network
    specs_file = results_folder + "backup/specs.json"
    import json
    with open(specs_file) as specs:
        specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
        specs = json.loads(specs)
    mixture_latent_size = specs["MixtureCodeLength"]
    patch_latent_size = specs["PatchCodeLength"]
    if specs["Encoder"] == "True":
        from deep_sdf.workspace import read_patch_encoder_param
        patch_encoder = read_patch_encoder_param(specs["PatchEncoderLayers"])
    else:
        patch_encoder = None
    network = Network(patch_latent_size=patch_latent_size, 
                        mixture_latent_size=mixture_latent_size, 
                        patch_encoder=patch_encoder, 
                        script_mode=True,
                        **specs["NetworkSpecs"])
    network = network.cuda()

    def load_weights_into_network(network, checkpoint=None, path=None):
        if path is not None and checkpoint is not None:
            raise RuntimeError("trying to load weights from two sources")
        if checkpoint is not None:
            path = results_folder + "ModelParameters/" + checkpoint + ".pth"
        weights = torch.load(path)
        model_state_dict = weights["model_state_dict"]
        prefix_len = len("module.")
        model_state_dict = {k[prefix_len:]:v for k,v in model_state_dict.items()} # network = torch.nn.Parallel(network) is done during training, which adds "module." to the front
        network.load_state_dict(model_state_dict)
        return network

    load_weights_into_network(network, checkpoint=checkpoint)
    network.eval()

    from reconstruct import reconstruct as deepsdf_sdf_to_latent
    def sdf_to_latent(sdf, full_loss, latent_init=None, num_samples=8000, num_iterations=800, sdf_filename=None, lr=0.01, adjust_lr_every=None, decrease_lr_by=None, l2reg=True, mixture_latent_mode=None):
        if latent_init is not None:
            latent_init = latent_init.detach().clone()
        error, latent = deepsdf_sdf_to_latent(network, 
                            num_iterations=num_iterations, 
                            latent_size=mixture_latent_size,
                            test_sdf=sdf,
                            stat=0.01,
                            clamp_dist=0.1,
                            num_samples=num_samples,
                            lr=lr,
                            l2reg=l2reg,
                            latent=latent_init,
                            full_loss=full_loss,
                            specs=specs,
                            sdf_filename=sdf_filename,
                            adjust_lr_every=None,
                            decrease_lr_by=None,
                            mixture_latent_mode=mixture_latent_mode
                            )
        return error, latent

    def patch_forward(latent, xyz, ode_evaluation_times=[0., 1.], ode_return_pos=False):
        latent = latent.reshape(1, -1)
        num_samples = xyz.shape[0]
        latent_repeat = latent.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, xyz], 1)

        return network.patch_network_forward(inputs, ode_evaluation_times=ode_evaluation_times, ode_return_pos=ode_return_pos)

    from deep_sdf.mesh import create_mesh as deepsdf_latent_to_mesh
    def latent_to_mesh(latent, patch_only=False, grid_resolution=256):
        max_batch = int(2**16) 
        if hasattr(network, "num_patches") and not patch_only:
            max_batch /= network.num_patches
        vertices, faces = deepsdf_latent_to_mesh(network,
                                latent,
                                "",
                                max_batch=int(max_batch),
                                file_format=None,
                                decoder_forward=patch_forward if patch_only else None,
                                N=grid_resolution
                                )
        return vertices, faces

    def get_training_latent_vectors(checkpoint="latest"):
        training_latent_vectors = results_folder + "LatentCodes/" + checkpoint + ".pth"
        npyfilenames = results_folder + "backup/special_files/npyfilenames.txt"
        
        with open(npyfilenames, "r") as npyfilenames:
            npyfilenames = npyfilenames.readlines()#[:-1]
            npyfilenames = [file[:-1] for file in npyfilenames]
        training_latent_vectors = torch.load(training_latent_vectors)["latent_codes"].detach().numpy()

        filename_to_latent = dict([(filename, latent[0]) for filename, latent in zip(npyfilenames, training_latent_vectors)])
        return filename_to_latent

    sys.path.remove(extra_sys_folder)

    return network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs
    
def create_folder(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def _read_mesh(mesh_file, use_vertices=True, use_faces=True, use_color=False, return_list_of_faces=False, prealloc_vertices=None, return_asap=False, meshlab=False, skip_adjacency_and_num_neighbors=False):
    vertex_pos = [(0., 0., 0.)]
    adjacency = np.zeros((1,1), dtype=np.int64)
    num_neighbors = np.zeros((1), dtype=np.int64)
    if use_color:
        colors = [(0., 0., 0.)]
    if meshlab:
        vertex_normals = [(0., 0., 0.)]
    list_of_faces = []
    current_vertex = 1
    if mesh_file[-4:] == ".obj":
        with open(mesh_file, "r") as mesh_file:
            for line in mesh_file:
                if use_vertices and line[3:] == "vn ":
                    new_normal = [float(x) for x in line[2:].split(" ")]
                    vertex_normals.append(new_normal)
                if use_vertices and line[:2] == "v ":
                    new_pos = [float(x) for x in line[2:].split(" ")[:3]]
                    vertex_pos.append(new_pos)
                    if use_color:
                        new_color = [float(x) for x in line[2:].split(" ")[3:]]
                        colors.append(new_color)
                    current_vertex += 1
                if line[:2] == "f ":
                    if use_faces:
                        if meshlab:
                            face = [int(x.split("/")[0]) for x in line[2:].split(" ")]
                        else:
                            face = [int(x) for x in line[2:].split(" ")]
                        if not skip_adjacency_and_num_neighbors:
                            adjacency, num_neighbors = _add_face(face, adjacency, num_neighbors)
                        if return_list_of_faces:
                            list_of_faces.append(face)
                    else:
                        if return_asap: # assume all vertices occur before all faces occur in the file
                            break
    vertex_pos = np.array(vertex_pos, dtype=np.float32)
    additional_returns = []
    if use_color:
        additional_returns.append(colors)
    if meshlab:
        vertex_normals = np.array(vertex_normals, dtype=np.float32)
        additional_returns.append(vertex_normals)
    if return_list_of_faces:
        if additional_returns:
            return vertex_pos, adjacency, num_neighbors, np.array(list_of_faces, dtype=np.int64), additional_returns
        else:
            return vertex_pos, adjacency, num_neighbors, np.array(list_of_faces, dtype=np.int64)
    else:
        if additional_returns:
            return vertex_pos, adjacency, num_neighbors, additional_returns
        else:
            return vertex_pos, adjacency, num_neighbors


def reconstruct_partial_shape(
    decoder,
    num_iterations,
    test_sdf,
    latent,
    freespace_sdf_value,
    num_samples,
    lr,
    keep_metadata_fixed=False,
    l2reg=False,
    specs=None,
    depth_map_losses=False,
    adjust_lr_every=None,
    decrease_lr_by=None,
    mixture_latent_mode=None,
    stat=0.01,
    clamp_dist=0.1
):
    import torch
    
    def unpack_sdf_samples_from_ram(data, subsample=None):
        if subsample is None:
            return data
        pos_tensor = data[0]
        neg_tensor = data[1]

        pos_size = pos_tensor.shape[0]
        neg_size = neg_tensor.shape[0]

        # split the sample into half
        half = int(subsample / 2)

        if pos_size <= half:
            random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
            sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        else:
            pos_start_ind = random.randint(0, pos_size - half)
            sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

        if neg_size <= half:
            random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        else:
            neg_start_ind = random.randint(0, neg_size - half)
            sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

        samples = torch.cat([sample_pos, sample_neg], 0)

        return samples

    latent = latent.detach().clone()
    
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decrease_lr_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decrease_lr_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    if decrease_lr_by is None:
        decrease_lr_by = 2
    if adjust_lr_every is None:
        adjust_lr_every = 100 #max(int(num_iterations / 3), 1)

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    decoder.do_code_regularization = l2reg

    loss_num = 0
    loss_l1 = torch.nn.L1Loss(reduction="sum")

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        xyz = xyz.reshape(1, num_samples, 3)
        sdf_gt = sdf_gt.reshape(1, num_samples, 1)
        
        freespace_samples = sdf_gt.view(-1) == freespace_sdf_value # shape: samples
            
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decrease_lr_by, adjust_lr_every)

        optimizer.zero_grad()

        inputs = {}
        inputs["mixture_latent_vectors"] = latent.view(1, -1) # 1 x mixture_latent_size
        inputs["xyz"] = xyz
        inputs["use_patch_encoder"] = False
        inputs["num_samp_per_scene"] = num_samples
        inputs["sdf_gt"] = sdf_gt
        inputs["mixture_latent_mode"] = specs["NetworkSpecs"]["mixture_latent_mode"] if mixture_latent_mode is None else mixture_latent_mode

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        #if e == 0:
        #    pred_sdf = decoder(inputs)

        if type(pred_sdf) == tuple:
            pred_sdf, extra_losses = pred_sdf

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = torch.zeros(1).cuda()[0]
        if depth_map_losses:
            if decoder.num_patches == 1:
                patch_sdfs = pred_sdf.view(-1, 1)
                
                extra_losses["freespace"] = torch.sum(torch.abs(torch.clamp(patch_sdfs[freespace_samples,:], max=0.))) / num_samples
                
                main_loss_weight = 1.0
                loss += main_loss_weight * loss_l1(patch_sdfs[~freespace_samples,0], inputs["sdf_gt"][0,~freespace_samples,0]) / num_samples
            else:
                patch_sdfs = decoder.patch_network_sdfs # samples x num_patches
                patch_weights = decoder.patch_network_mixture_weights # samples x num_patches
                
                patch_mask = patch_weights == 0.

                patch_recon = torch.zeros_like(patch_sdfs) # samples x num_patches
                if torch.any(~freespace_samples):
                    patch_recon[~freespace_samples,:] = patch_sdfs[~freespace_samples,:] - inputs["sdf_gt"][0,~freespace_samples,0].view(-1, 1) # broadcast across patches
                if torch.any(freespace_samples):
                    patch_recon[freespace_samples,:] = torch.clamp(patch_sdfs[freespace_samples,:], max=0.)
                patch_recon[patch_mask] = 0.
                patch_recon = patch_recon.view(-1, inputs["num_samp_per_scene"], decoder.num_patches)
                
                patch_recon = torch.abs(patch_recon)
                patch_recon = patch_recon / (torch.sum((~patch_mask).view(-1, inputs["num_samp_per_scene"], decoder.num_patches), dim=1).unsqueeze(1).float() + 0.000001)
                direct_patch_loss = torch.sum(patch_recon, dim=1)
                direct_patch_weight = 10.
                direct_patch_loss = direct_patch_weight * torch.mean(direct_patch_loss)
                extra_losses["direct_patch"] = direct_patch_loss

        for extra_loss_name, extra_loss_value in extra_losses.items():
            if "latent_regularization" in extra_loss_name:
                loss += 1e-4 * extra_loss_value
            else:
                loss += extra_loss_value

        loss.backward()

        if keep_metadata_fixed:
            for i in range(decoder.num_patches):
                patch_latent_size = decoder.patch_latent_size
                offset = i * (patch_latent_size + 7)
                latent.grad.data[0, offset + patch_latent_size : offset + patch_latent_size + 7] = 0.
        
        optimizer.step()
    
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def _partial_chamfer_distance(partial_groundtruth_point_cloud_file, groundtruth_mesh_file, groundtruth_npz, regressed_mesh_file, num_points=100000):

    from im2mesh.eval import distance_p2p
    import trimesh

    partial_groundtruth_surface_samples = _read_mesh(partial_groundtruth_point_cloud_file)[0]
    
    full_groundtruth_surface_samples = precompute_surface_samples(groundtruth_mesh_file, num_points)

    regressed_trimesh = trimesh.load(regressed_mesh_file, process=False)
    
    x,y,z,scale = np.load(groundtruth_npz)["normalization_parameters"]
    regressed_trimesh.vertices *= scale
    regressed_trimesh.vertices += np.array([[x[0],y[0],z[0]]])
    partial_groundtruth_surface_samples *= scale
    partial_groundtruth_surface_samples += np.array([[x[0],y[0],z[0]]])
    
    regressed_surface_samples = regressed_trimesh.sample(num_points)

    # distance_p2p(x, None, y, None): compute distance from each point in x to its closest point in y
    completeness, _ = distance_p2p(partial_groundtruth_surface_samples, None, regressed_surface_samples, None) # shape: num_points
    accuracy, _ = distance_p2p(regressed_surface_samples, None, full_groundtruth_surface_samples, None)
    
    full_to_regressed_distances, _ = distance_p2p(full_groundtruth_surface_samples, None, regressed_surface_samples, None)
    
    regressed_to_full = accuracy
    mesh_accuracy_deepsdf = np.percentile(regressed_to_full, 90)

    f_score_threshold = 0.01 # deep structured implicit functions sets tau = 0.01

    partial_to_regressed = 100. * (completeness**2).mean()
    regressed_to_full = 100. * (accuracy**2).mean()
    full_to_regressed = 100. * (full_to_regressed_distances**2).mean()
    # L1 chamfer
    l1_chamfer = 100. * (completeness.mean() + accuracy.mean())
    # L2 chamfer
    l2_chamfer = 100. * ((completeness**2).mean() + (accuracy**2).mean())
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = 100. * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy) # harmonic mean
    f_full_to_regressed = np.mean(full_to_regressed_distances <= f_score_threshold)
    f_score_full = 100. * 2 * f_full_to_regressed * f_accuracy / (f_full_to_regressed + f_accuracy) # harmonic mean

    return l1_chamfer, l2_chamfer, f_score, partial_to_regressed, regressed_to_full, f_completeness, f_accuracy, f_full_to_regressed, full_to_regressed, mesh_accuracy_deepsdf, f_score_full


def _sdf_to_rgb(sdf, invalid_value=+2):
    valid_entries = sdf != invalid_value
    valid_sdf = sdf[valid_entries]
    min_sdf = np.min(valid_sdf)
    max_sdf = np.max(valid_sdf)
    absolute_scale_sdf = max(abs(min_sdf), abs(max_sdf))
    rgb = np.zeros((sdf.shape[0], 3))
    mask = sdf < 0
    rgb[mask, 0] = np.sqrt(-sdf[mask] / absolute_scale_sdf) # red
    rgb[~mask, 1] = np.sqrt(sdf[~mask] / absolute_scale_sdf) # green
    rgb[~valid_entries,:] = np.array([0,0,1]) # blue
    return rgb



def shape_completion_deepsdf():

    import torch
    
    results_folder = "/" # the results folder that was used to train the network, not the test results folder
        
    depthmode = "frontal" # halfplane, frontal ("fixed" in the paper), random0 ("random" in the paper), random1
    
    checkpoint = "latest"
    
    # for shape completion, set evaluate = only_evaluate = False
    # to evaluate the obtained shape completion, set evaluate = only_evaluate = True. This mode needs to be run in the same way as the evaluation mode in evaluate_patch_network_metrics()
    evaluate = False
    only_evaluate = False
    
    #mesh_files = "examples/splits/sv1_airplanes_50_test.json"
    mesh_files = "examples/splits/sv1_sofa_50_test.json"
    data_source = "shapenetv1/deepsdf_preprocessed/SdfSamples/"
    groundtruth_meshes_folder = "shapenetv1/watertight/"

    use_full_data_for_recon = False # do not change
    if not use_full_data_for_recon:    
        local_refinement = False # "refined" in the paper. Only use for PatchNets, not for DeepSFD or baseline.
        use_special_reconstruct_function = True # do not change
    
    ### the parameters below can be changed but are set to the values used to obtain the results in the paper
    
    # projection matrix. values are s_cam2.GetProjectionModelViewMatrix() from PreprocessMesh.cpp from DeepSDF preprocessing. only used for "frontal" depth_mode.
    model_view = np.array([[-0.707107,  0,        -0.707107,  0], \
                           [-0.408248,  0.816497,  0.408248,  0], \
                           [ 0.57735,   0.57735,  -0.57735,  -1.73205], \
                           [ 0,         0,         0,         1]]) # corner view for ShapeNet

    seed = 3254

    invalid_depth_value = 2.
    freespace_sdf_value = 3.
    generate_sdf_surface_distance = 0.005
    freespace_points_ratio = 0.3

    keep_metadata_fixed_during_refinement = True
    only_keep_coverage_and_recon_loss = True
    full_loss = True
    use_l2_regularization_for_local_refinement = False
    use_l2_regularization = True
    num_samples = 8000
    num_iterations = 600
    lr = 0.01
    refinement_lr = 0.001
    refinement_num_iterations = 100
    adjust_lr_every = 200
    decrease_lr_by = 2
    grid_resolution = 128
    output_name = "shape_completion"
   
    if checkpoint is None:
        checkpoint = "latest"
    output_name += "_" + depthmode
    output_name += "/"
    output_folder = results_folder + "output/parts/" + output_name

    create_folder(output_folder)
    
    if not use_full_data_for_recon and local_refinement:
        num_iterations -= refinement_num_iterations
    
    if only_evaluate:
        evaluate = True

    if not only_evaluate:
        network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs = _setup_torch_network(results_folder)
        load_weights_into_network(network, checkpoint=checkpoint)
    
        if only_keep_coverage_and_recon_loss:
            network.pull_free_space_patches_to_surface = False
            network.align_patch_rotation_with_normal = False
            network.keep_scales_small = False
            network.scales_low_variance = False

    import json
    if mesh_files[-5:] == ".json":
        with open(mesh_files, "r") as mesh_files:
            mesh_files = list(json.load(mesh_files).items())
        for super_folder, entries in mesh_files:
            for subfolder, subentries in entries.items():
                mesh_files = [super_folder + "/" + subfolder + "/" + mesh_file for mesh_file in subentries]
    else:
        with open(mesh_files, "r") as mesh_files:
            mesh_files = mesh_files.readlines()[:-1]

    # determine latent codes of known meshes
    train_latent_codes = torch.load(results_folder + "LatentCodes/" + checkpoint + ".pth")
    known_latent_codes = train_latent_codes["latent_codes"]
    known_latent_codes = known_latent_codes.squeeze(1)
    means = torch.mean(known_latent_codes, dim=0).detach().numpy() # shape: latent_size
    
    for i, mesh in enumerate(mesh_files):
    
        if evaluate and only_evaluate:
            break

        print(str(i) + " " + mesh, flush=True)

        if "____rejected.npz" in mesh:
            mesh = mesh[:-len("____rejected.npz")]

        latent_code = torch.from_numpy(means).reshape(1,-1).float().cuda()

        sdf_filename = data_source + mesh + ".npz"
        try:
            sdf = np.load(sdf_filename)
        except:
            sdf = np.load(sdf_filename + "____rejected.npz")
            
        if depthmode != "halfplane" and not use_full_data_for_recon:
            # assumes orthogonal projection. x and y components range from [-1,+1]. z component is in world coordinates (not distance to camera!).
            # this works off of OpenGL. so the values of the depth map are also in [-1,+1]. they are in rotated world coordinates (rotation according to model_view).
            
            if depthmode == "random0":
                rotated_world_depth_image = sdf["random_depth_0"] # height x width
                model_view = sdf["random_depth_0_modelview"] # 4 x 4
                if np.all(rotated_world_depth_image == invalid_depth_value):
                    rotated_world_depth_image = sdf["random_depth_1"] # height x width
                    model_view = sdf["random_depth_1_modelview"] # 4 x 4
            elif depthmode == "random1":
                rotated_world_depth_image = sdf["random_depth_1"] # height x width
                model_view = sdf["random_depth_1_modelview"] # 4 x 4
            elif depthmode == "frontal":
                rotated_world_depth_image = sdf["depth"] # height x width
   
            height, width = rotated_world_depth_image.shape
         
            # rotated depth map to canonical point cloud
            height_normalized_coordinates = -np.linspace(-1., +1., num=height)
            width_normalized_coordinates = np.linspace(-1., +1., num=width)
            height_normalized_coordinates_image = np.repeat( height_normalized_coordinates.reshape(1, -1), width, axis=0).transpose()
            width_normalized_coordinates_image = np.repeat( width_normalized_coordinates.reshape(1, -1), height, axis=0)
            ones = np.ones_like(width_normalized_coordinates_image)
                        
            normalized_depth_image = rotated_world_depth_image.copy()
            
            full_normalized_coordinates_image = np.stack([width_normalized_coordinates_image, height_normalized_coordinates_image, normalized_depth_image, ones], axis=-1) # height x width x 4 (x_n, y_n, d_n, 1)
            world_coordinates = np.matmul(np.linalg.inv(model_view), np.expand_dims(full_normalized_coordinates_image, axis=-1)).squeeze(axis=-1)[:,:,:3] 
            
            # rotated depth map to rotated normals
            # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
            spacing = 2./(height-1)
            zy, zx = np.gradient(rotated_world_depth_image, spacing)
            normal = np.dstack((-zx, zy, np.ones_like(rotated_world_depth_image))) # need to flip zy because OpenGL indexes bottom left as (0,0) (this is a guess, it simply turns out to work if zy is flipped).
            normal_length = np.linalg.norm(normal, axis=2, keepdims=True)
            normal /= normal_length         
            
            valid_depth = rotated_world_depth_image != invalid_depth_value
            view_alignment = 0.1
            valid_normals = normal[:,:,2] > view_alignment
            
            # rotate normals to canonical
            normal = np.matmul( np.linalg.inv(model_view[:3,:3]) , np.expand_dims(normal, axis=-1)).squeeze(-1) # height x width x 3
        
            # get valid canonical points with normals
            valid_pixels = valid_depth.astype(np.bool) & valid_normals.astype(np.bool)
            valid_pixels = valid_pixels.reshape(-1)
            
            surface_points = world_coordinates.reshape(-1, 3)[valid_pixels,:] # num_points x 3
            surface_normals = normal.reshape(-1, 3)[valid_pixels,:] # num_points x 3
                    
            # generate SDF points
            pos_samples = surface_points + generate_sdf_surface_distance * surface_normals
            neg_samples = surface_points - generate_sdf_surface_distance * surface_normals
            
            pos_samples = np.concatenate([pos_samples, generate_sdf_surface_distance * np.ones((pos_samples.shape[0], 1))], axis=-1)
            neg_samples = np.concatenate([neg_samples, -generate_sdf_surface_distance * np.ones((pos_samples.shape[0], 1))], axis=-1)
        
            # freespace points
            freespace = full_normalized_coordinates_image.copy().reshape(-1, 4)

            close_freespace = freespace[valid_depth.flatten(),:].copy()
            close_freespace[:,2] += np.abs(np.random.normal(0., .03, size=(np.sum(valid_depth))))   
            freespace[~valid_depth.flatten(),2] -= invalid_depth_value - model_view[2,3] # pull background to the center of the shape
            freespace[~valid_depth.flatten(),2] += np.random.uniform(-1., 1., size=(np.sum(~valid_depth)))
            freespace[valid_depth.flatten(),2] += np.random.uniform(0., 1.5, size=(np.sum(valid_depth)))  
            
            freespace = np.concatenate([freespace, close_freespace], axis=0)
            freespace = np.matmul(np.linalg.inv(model_view), np.expand_dims(freespace, axis=-1)).squeeze(axis=-1)[:,:3]
            freespace = np.concatenate([freespace, freespace_sdf_value * np.ones((freespace.shape[0], 1))], axis=1)
            
            num_freespace_points = int(freespace_points_ratio * 2 * surface_points.shape[0])
            freespace_indices = np.random.choice(freespace.shape[0], num_freespace_points)
            freespace = freespace[freespace_indices,:]
            
            pos_samples = np.concatenate([pos_samples, freespace[:num_freespace_points//2,:]], axis=0)
            neg_samples = np.concatenate([neg_samples, freespace[num_freespace_points//2:,:]], axis=0)
        
            reduced_sdf = (pos_samples.astype(np.float32), neg_samples.astype(np.float32))
        
        else:
            sdf = (sdf["pos"], sdf["neg"])

            def _make_noisy(sdf, seed):
                pos_sdf = sdf[0]
                neg_sdf = sdf[1]
                rotation = scipy.spatial.transform.Rotation.random(random_state=seed)
                pos_sdf[:,:3] = rotation.apply(pos_sdf[:,:3])
                neg_sdf[:,:3] = rotation.apply(neg_sdf[:,:3])
                reduced_pos_sdf = pos_sdf[pos_sdf[:,0] >= 0, :]
                reduced_neg_sdf = neg_sdf[neg_sdf[:,0] >= 0, :]
                reduced_pos_sdf[:,:3] = rotation.apply(reduced_pos_sdf[:,:3], inverse=True)
                reduced_neg_sdf[:,:3] = rotation.apply(reduced_neg_sdf[:,:3], inverse=True)
                return (reduced_pos_sdf, reduced_neg_sdf)
            reduced_sdf = sdf if use_full_data_for_recon else _make_noisy(sdf, seed)
            seed += 234
            
            surface_sdf_threshold = 0.01
            intermediate = np.concatenate([reduced_sdf[0], reduced_sdf[1]], axis=0)
            surface_points = intermediate[ np.abs(intermediate[:,3]) < surface_sdf_threshold, :3]

        mesh_folder = os.path.split(output_folder + mesh)[0]
        create_folder(mesh_folder)

        rgb = _sdf_to_rgb(np.concatenate([reduced_sdf[0][:,-1], reduced_sdf[1][:,-1]], axis=0))
        _write_mesh(np.concatenate([reduced_sdf[0][:,:3], reduced_sdf[1][:,:3]], axis=0), None, output_folder + mesh + "_reduced_sdf.obj", color=rgb)
        
        _write_mesh(surface_points, None, output_folder + mesh + "_partial_surface_point_cloud.obj") # keep this! needed for evaluation

        reduced_sdf = (torch.from_numpy(reduced_sdf[0]), torch.from_numpy(reduced_sdf[1]))
        if use_special_reconstruct_function:
            _, latent_code = reconstruct_partial_shape(network, num_iterations=num_iterations, test_sdf=reduced_sdf, latent=latent_code, freespace_sdf_value=freespace_sdf_value, num_samples=num_samples, lr=lr, depth_map_losses=True, adjust_lr_every=adjust_lr_every, decrease_lr_by=decrease_lr_by, l2reg=use_l2_regularization, specs=specs, mixture_latent_mode=None)
        else:
            _, latent_code = sdf_to_latent(reduced_sdf, full_loss=full_loss, latent_init=latent_code, num_samples=num_samples, num_iterations=num_iterations, sdf_filename=sdf_filename, lr=lr, adjust_lr_every=adjust_lr_every, decrease_lr_by=decrease_lr_by, l2reg=use_l2_regularization)
        
        if local_refinement: # needs to be right after sdf_to_latent to get the metadata from the network's script_mode
            def _get_current_explicit_latent_vector():
                latents = network.patch_latent_vectors.detach().clone().squeeze(0).cpu().numpy() # num_patches x patch_latent_size
                positions = network.patch_positions.detach().clone().squeeze(0).cpu().numpy() # num_patches x 3
                rotations = network.patch_rotations.detach().clone().squeeze(0).cpu().numpy() # num_patches x 3
                scalings = network.patch_scalings.detach().clone().squeeze(0).cpu().numpy() # num_patches x 1
                refined_latent_code = np.concatenate([ np.concatenate([lat, pos, rot, sca], axis=0) for lat, pos, rot, sca in zip(latents, positions, rotations, scalings) ], axis=0)
                refined_latent_code = torch.from_numpy(refined_latent_code).reshape(1,-1).float().cuda()
                return refined_latent_code
            refined_latent_code = _get_current_explicit_latent_vector()
            implicit_latent_size = network.mixture_latent_size
            network.mixture_latent_size = refined_latent_code.shape[1]
            if use_special_reconstruct_function:
                _, refined_latent_code = reconstruct_partial_shape(network, num_iterations=refinement_num_iterations, test_sdf=reduced_sdf, latent=refined_latent_code, freespace_sdf_value=freespace_sdf_value, num_samples=num_samples, lr=refinement_lr, depth_map_losses=True, adjust_lr_every=adjust_lr_every, decrease_lr_by=decrease_lr_by, l2reg=use_l2_regularization_for_local_refinement, specs=specs, mixture_latent_mode="all_explicit", keep_metadata_fixed=keep_metadata_fixed_during_refinement)
            else:
                _, refined_latent_code = sdf_to_latent(reduced_sdf, full_loss=full_loss, latent_init=refined_latent_code, num_samples=num_samples, num_iterations=refinement_num_iterations, sdf_filename=sdf_filename, lr=refinement_lr, adjust_lr_every=adjust_lr_every, decrease_lr_by=decrease_lr_by, l2reg=use_l2_regularization_for_local_refinement,\
                    mixture_latent_mode="all_explicit")
            network.mixture_latent_mode = "all_explicit" # latent_to_mesh needs correct mixture latent mode
            vertices, faces = latent_to_mesh(refined_latent_code, grid_resolution=grid_resolution)
            network.mixture_latent_mode = "all_implicit"      
            network.mixture_latent_size = implicit_latent_size
            
            _write_mesh(vertices, faces+1, output_folder + mesh + "_completed_refined.obj")
            torch.save( {"epoch": None, "latent_codes": refined_latent_code.reshape(1,1,-1)}, output_folder + mesh + "_completed_refined_latent.pth")
            
        vertices, faces = latent_to_mesh(latent_code, grid_resolution=grid_resolution)
        
        _write_mesh(vertices, faces+1, output_folder + mesh + "_completed.obj")
        torch.save( {"epoch": None, "latent_codes": latent_code.reshape(1,1,-1)}, output_folder + mesh + "_completed_latent.pth")
        
    if evaluate:
        unrefined_results = {}
        refined_results = {}

        for i, mesh in enumerate(mesh_files):
        
            partial_groundtruth_point_cloud_file = output_folder + mesh + "_partial_surface_point_cloud.obj"
            groundtruth_mesh_file = groundtruth_meshes_folder + mesh[len(super_folder)+1:]
            groundtruth_npz = data_source + mesh + ".npz"
        
            # unrefined
            regressed_mesh_file = output_folder + mesh + "_completed.obj"
            l1_chamfer, l2_chamfer, f_score, partial_to_regressed, regressed_to_full, f_completeness, f_accuracy, f_full_to_regressed, full_to_regressed, mesh_accuracy_deepsdf, f_score_full = _partial_chamfer_distance(partial_groundtruth_point_cloud_file, groundtruth_mesh_file, groundtruth_npz, regressed_mesh_file)
            unrefined_results[mesh] = {"L1_chamfer": l1_chamfer, "L2_chamfer": l2_chamfer, "f_score": f_score, "partial_to_regressed": partial_to_regressed, "regressed_to_full": regressed_to_full, "f_completeness": f_completeness, "f_accuracy": f_accuracy, "f_full_to_regressed": f_full_to_regressed, "full_to_regressed": full_to_regressed, "mesh_accuracy_deepsdf": mesh_accuracy_deepsdf, "f_score_full": f_score_full}
            
            # refined
            regressed_mesh_file = output_folder + mesh + "_completed_refined.obj"
            if os.path.exists(regressed_mesh_file):
                l1_chamfer_refined, l2_chamfer_refined, f_score_refined, partial_to_regressed_refined, regressed_to_full_refined, f_completeness_refined, f_accuracy_refined, f_full_to_regressed_refined, full_to_regressed_refined, mesh_accuracy_deepsdf_refined, f_score_full_refined = _partial_chamfer_distance(partial_groundtruth_point_cloud_file, groundtruth_mesh_file, groundtruth_npz, regressed_mesh_file)
                refined_results[mesh] = {"L1_chamfer": l1_chamfer_refined, "L2_chamfer": l2_chamfer_refined, "f_score": f_score_refined, "partial_to_regressed": partial_to_regressed_refined, "regressed_to_full": regressed_to_full_refined, "f_completeness": f_completeness_refined, "f_accuracy": f_accuracy_refined, "f_full_to_regressed": f_full_to_regressed_refined, "full_to_regressed": full_to_regressed_refined, "mesh_accuracy_deepsdf": mesh_accuracy_deepsdf_refined, "f_score_full": f_score_full_refined}

        import json
        with open(output_folder + "individual_results_unrefined.json", "w") as output_json:
            json.dump(unrefined_results, output_json, indent=2)
        with open(output_folder + "individual_results_refined.json", "w") as output_json:
            json.dump(refined_results, output_json, indent=2)
            
        average_results_unrefined = {}
        average_results_unrefined["L1_chamfer"] = np.mean([result["L1_chamfer"] for result in unrefined_results.values()])
        average_results_unrefined["L2_chamfer"] = np.mean([result["L2_chamfer"] for result in unrefined_results.values()])
        average_results_unrefined["f_score"] = np.mean([result["f_score"] for result in unrefined_results.values()])
        average_results_unrefined["partial_to_regressed"] = np.mean([result["partial_to_regressed"] for result in unrefined_results.values()])
        average_results_unrefined["regressed_to_full"] = np.mean([result["regressed_to_full"] for result in unrefined_results.values()])
        average_results_unrefined["f_completeness"] = np.mean([result["f_completeness"] for result in unrefined_results.values()])
        average_results_unrefined["f_accuracy"] = np.mean([result["f_accuracy"] for result in unrefined_results.values()])
        average_results_unrefined["f_full_to_regressed"] = np.mean([result["f_full_to_regressed"] for result in unrefined_results.values()])
        average_results_unrefined["full_to_regressed"] = np.mean([result["full_to_regressed"] for result in unrefined_results.values()])
        average_results_unrefined["mesh_accuracy_deepsdf"] = np.mean([result["mesh_accuracy_deepsdf"] for result in unrefined_results.values()])
        average_results_unrefined["f_score_full"] = np.mean([result["f_score_full"] for result in unrefined_results.values()])
        
        with open(output_folder + "average_results_unrefined.json", "w") as output_json:
            json.dump(average_results_unrefined, output_json, indent=2)

        average_results_refined = {}
        average_results_refined["L1_chamfer"] = np.mean([result["L1_chamfer"] for result in refined_results.values()])
        average_results_refined["L2_chamfer"] = np.mean([result["L2_chamfer"] for result in refined_results.values()])
        average_results_refined["f_score"] = np.mean([result["f_score"] for result in refined_results.values()])
        average_results_refined["partial_to_regressed"] = np.mean([result["partial_to_regressed"] for result in refined_results.values()])
        average_results_refined["regressed_to_full"] = np.mean([result["regressed_to_full"] for result in refined_results.values()])
        average_results_refined["f_completeness"] = np.mean([result["f_completeness"] for result in refined_results.values()])
        average_results_refined["f_accuracy"] = np.mean([result["f_accuracy"] for result in refined_results.values()])
        average_results_refined["f_full_to_regressed"] = np.mean([result["f_full_to_regressed"] for result in refined_results.values()])
        average_results_refined["full_to_regressed"] = np.mean([result["full_to_regressed"] for result in refined_results.values()])
        average_results_refined["mesh_accuracy_deepsdf"] = np.mean([result["mesh_accuracy_deepsdf"] for result in refined_results.values()])
        average_results_refined["f_score_full"] = np.mean([result["f_score_full"] for result in refined_results.values()])

        with open(output_folder + "average_results_refined.json", "w") as output_json:
            json.dump(average_results_refined, output_json, indent=2)

def _convert_euler_to_matrix(angles): 
    # angles: N x 3
    sine = np.sin(angles)
    cosine = np.cos(angles)
        
    sin_alpha, sin_beta, sin_gamma = sine[:,0], sine[:,1], sine[:,2]
    cos_alpha, cos_beta, cos_gamma = cosine[:,0], cosine[:,1], cosine[:,2]

    R00 = cos_gamma*cos_beta;
    R01 = -sin_gamma*cos_alpha + cos_gamma*sin_beta*sin_alpha;
    R02 = sin_gamma*sin_alpha + cos_gamma*sin_beta*cos_alpha;
    
    R10 = sin_gamma*cos_beta;
    R11 = cos_gamma*cos_alpha + sin_gamma*sin_beta*sin_alpha;
    R12 = -cos_gamma*sin_alpha + sin_gamma*sin_beta*cos_alpha;
    
    R20 = -sin_beta;
    R21 = cos_beta*sin_alpha;
    R22 = cos_beta*cos_alpha;

    R0 = np.stack([R00, R01, R02], 1) # first row
    R1 = np.stack([R10, R11, R12], 1) # second row
    R2 = np.stack([R20, R21, R22], 1) # third row

    R = np.stack([R0, R1, R2], 1) # shape: (batch_size, row, column)
    return R
    
    
def _cleanup_vertices_faces(vertices, faces, clamping_distance, patch_is_centered, patch_center=None, colors=None):

    if not patch_is_centered:
        vertices -= np.expand_dims(patch_center, axis=0)

    distances = np.linalg.norm(vertices, axis=1)
    good_vertices = distances <= clamping_distance
    good_indices = np.where(good_vertices)

    invalid_index = -1
    new_vertex_ids = np.zeros((vertices.shape[0]), dtype=int) + invalid_index
    new_vertex_ids[good_indices] = np.arange(np.sum(good_vertices))
    faces = new_vertex_ids[faces.flat].reshape(-1, 3)
    good_faces = np.all(faces != invalid_index, axis=1)

    vertices = vertices[good_vertices,:]
    if colors is not None and colors.size > 0:
        colors = colors[good_vertices,:]
    faces = faces[good_faces, :]

    if colors is None:
        return vertices, faces
    else:
        return vertices, faces, colors
    

def visualize_parts_individually(results_folder=None, grid_resolution=None, checkpoint=None):

    import torch

    if results_folder is None:
        results_folder = "/"
    if checkpoint is None:
        checkpoint = None
    if grid_resolution is None:
        grid_resolution = 64
    mesh_files = None
    data_source = None
    save_each_part_in_own_obj = False
    output_name = "parts_individually"

    if checkpoint is None:
        checkpoint = "latest"

    output_name += "_" + checkpoint + "_" + str(grid_resolution)
    output_name += "/"
    output_folder = results_folder + "output/parts/" + output_name
    create_folder(output_folder)
    backupThisFile(output_folder)

    network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs = _setup_torch_network(results_folder)
    network = load_weights_into_network(network, checkpoint=checkpoint)
    network.use_depth_encoder = False

    with open(results_folder + "backup/special_files/npyfilenames.txt", "r") as train_split:
        train_split = train_split.readlines()#[:-1]
        train_split = [line[:-5] for line in train_split] # remove ".npz\n"

    if mesh_files is None:
        mesh_files = train_split
    else:
        import json
        if mesh_files[-5:] == ".json":
            with open(mesh_files, "r") as mesh_files:
                mesh_files = list(json.load(mesh_files).items())
            for super_folder, entries in mesh_files:
                import ipdb; ipdb.set_trace()
                for subfolder, subentries in entries.items():
                    mesh_files = [super_folder + "/" + subfolder + "/" + mesh_file for mesh_file in subentries]
        else:
            with open(mesh_files, "r") as mesh_files:
                mesh_files = mesh_files.readlines()[:-1]

    def _guess_data_source():
        import json
        with open(results_folder + "backup/specs.json", "r") as specs:
            specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
            specs = json.loads(specs)
        data_source = specs["DataSource"] + "SdfSamples/"
        return data_source

    with open(output_folder + "MeshFiles", "w") as output_mesh_files:
        for mesh_file in mesh_files:
            output_mesh_files.write(mesh_file + ".obj\n")

    consistent_patch_colors = []

    train_latent_codes = torch.load(results_folder + "LatentCodes/" + checkpoint + ".pth")
    
    for i, mesh in enumerate(mesh_files):
        
        train_id = train_split.index(mesh)
        latent_code = train_latent_codes["latent_codes"][train_id,0,:].unsqueeze(0).clone().cuda()
        latent_code = latent_code[0]      

        network.align_patch_rotation_with_normal = False
        dummy_input = torch.cat([latent_code, torch.zeros((3)).cuda()], dim=0).unsqueeze(0)
        network(dummy_input)
        def copy_to_numpy(torch_array):
            return torch_array.detach().cpu().numpy()

        patch_latent_vectors = copy_to_numpy(network.patch_latent_vectors[0])
        patch_rotations = copy_to_numpy(network.patch_rotations[0])
        patch_positions = copy_to_numpy(network.patch_positions[0])
        if network.variable_patch_scaling:
            patch_scalings = copy_to_numpy(network.patch_scalings[0]) 
        else:
            patch_scalings = network.non_variable_patch_radius * np.ones((patch_positions.shape[0], 1))

        patch_rotations = _convert_euler_to_matrix(patch_rotations)

        vertices = np.zeros((0, 3), dtype=float)
        faces = np.zeros((0, 3), dtype=int)
        colors = np.zeros((0, 3), dtype=float)
        for j, (patch_latent, patch_rot, patch_pos, patch_scale) in enumerate(zip(patch_latent_vectors, patch_rotations, patch_positions, patch_scalings)):
            
            patch_latent = torch.from_numpy(patch_latent).cuda()

            patch_vertices, patch_faces = latent_to_mesh(patch_latent, patch_only=True, grid_resolution=grid_resolution)
            # remove vertices that are far away from the patch center, clean up faces accordingly
            patch_vertices, patch_faces = _cleanup_vertices_faces(patch_vertices, patch_faces, clamping_distance=1.0, patch_is_centered=True)

            patch_vertices *= patch_scale
            patch_vertices = np.matmul(patch_rot, patch_vertices.transpose()).transpose()
            patch_vertices += np.expand_dims(patch_pos, axis=0)

            try:
                patch_color = consistent_patch_colors[j]
            except:
                patch_color = np.random.rand(1, 3)
                consistent_patch_colors.append(patch_color)
            patch_colors = np.repeat(patch_color, patch_vertices.shape[0], axis=0)

            if save_each_part_in_own_obj:
                _write_mesh(patch_vertices, patch_faces+1, output_folder + "parts_object_" + str(i) + "_part_" + str(j) + ".obj", color=patch_colors)

            patch_faces += vertices.shape[0] # 0-indexed

            vertices = np.concatenate([vertices, patch_vertices], axis=0)
            faces = np.concatenate([faces, patch_faces], axis=0)
            colors = np.concatenate([colors, patch_colors], axis=0)

        mesh_folder = os.path.split(output_folder + mesh)[0]
        create_folder(mesh_folder)
        _write_mesh(vertices, faces+1, output_folder + mesh + ".obj", color=colors)
 
 

def _get_bounding_box(vertices):
    # axis-aligned bounding box
    return np.max(vertices, axis=0), np.min(vertices, axis=0)


def precompute_iou_on_watertight(groundtruth_mesh_file, num_points, use_precomputed_result=True):

    precomputed_file = groundtruth_mesh_file + "_iou_" + str(num_points) + ".npy"
    if os.path.exists(precomputed_file) and use_precomputed_result:
        watertight_result = np.load(precomputed_file) # num_points x 4 (x,y,z, 1.0f if inside else 0.0f)

    else:
        from im2mesh.utils.libmesh import check_mesh_contains
        import trimesh
        groundtruth_trimesh = trimesh.load(groundtruth_mesh_file, process=False)

        max_bb, min_bb = _get_bounding_box(groundtruth_trimesh.vertices)
        bb_samples = np.random.rand(num_points, 3)
        bb_samples = bb_samples * (max_bb - min_bb).reshape(1,3) + min_bb.reshape(1,3)

        inside = check_mesh_contains(groundtruth_trimesh, bb_samples) # shape: num_points

        watertight_result = np.concatenate([bb_samples, inside.reshape(-1,1).astype(float)], axis=1)

        np.save(precomputed_file, watertight_result)

    return watertight_result


def intersection_over_union(groundtruth_mesh_file, regressed_trimesh, num_points=100000):

    watertight_result = precompute_iou_on_watertight(groundtruth_mesh_file, num_points=num_points) # num_points x 4 (x,y,z, 1.0f if inside else 0.0f)

    from im2mesh.utils.libmesh import check_mesh_contains
    regressed_inside = check_mesh_contains(regressed_trimesh, watertight_result[:,:3]).astype(np.bool) # shape: num_points
    watertight_inside = watertight_result[:,3].astype(np.bool)

    area_union = (regressed_inside | watertight_inside).astype(np.float32).sum()
    area_intersect = (regressed_inside & watertight_inside).astype(np.float32).sum()

    iou = 100. * (area_intersect / area_union)

    return iou


def precompute_surface_samples(groundtruth_mesh_file, num_points, use_precomputed_samples=True):

    precomputed_file = groundtruth_mesh_file + "_surface_" + str(num_points) + ".npz"
    if os.path.exists(precomputed_file) and use_precomputed_samples:
        precomputed_dict = np.load(precomputed_file)
        groundtruth_surface_samples = precomputed_dict["samples"] # num_points x 3

    else:
        import trimesh
        groundtruth_trimesh = trimesh.load(groundtruth_mesh_file, process=False)
        groundtruth_surface_samples = groundtruth_trimesh.sample(num_points)

        np.savez(precomputed_file, samples=groundtruth_surface_samples)

    return groundtruth_surface_samples
    

def chamfer_distance(groundtruth_mesh_file, regressed_trimesh, num_points=100000):

    from im2mesh.eval import distance_p2p
    import trimesh

    groundtruth_surface_samples = precompute_surface_samples(groundtruth_mesh_file, num_points)

    regressed_surface_samples = regressed_trimesh.sample(num_points)

    completeness, _ = distance_p2p(groundtruth_surface_samples, None, regressed_surface_samples, None) # shape: num_points
    accuracy, _ = distance_p2p(regressed_surface_samples, None, groundtruth_surface_samples, None)

    f_score_threshold = 0.01 # deep structured implicit functions sets tau = 0.01

    # L1 chamfer
    l1_chamfer = 100. * (completeness.mean() + accuracy.mean())
    # L2 chamfer
    l2_chamfer = 100. * ((completeness**2).mean() + (accuracy**2).mean())
    # F-score
    f_completeness = np.mean(completeness <= f_score_threshold)
    f_accuracy = np.mean(accuracy <= f_score_threshold)
    f_score = 100. * 2 * f_completeness * f_accuracy / (f_completeness + f_accuracy) # harmonic mean

    return l1_chamfer, l2_chamfer, f_score


def _evaluate_object(args):

    object_, class_, regressed_meshes_folder, groundtruth_meshes_folder, dataset_name, data_source = args

    regressed_mesh_file = regressed_meshes_folder + dataset_name + "/" + class_ + "/" + object_ + ".obj"
    groundtruth_mesh_file = groundtruth_meshes_folder + class_ + "/" + object_
    groundtruth_npz = data_source + dataset_name + "/" + class_ + "/" + object_ + ".npz"

    import trimesh
    regressed_trimesh = trimesh.load(regressed_mesh_file, process=False)
    if type(regressed_trimesh) == list and len(regressed_trimesh) == 0: # empty mesh
        return {"iou": 0., "chamfer_l1": 100., "chamfer_l2": 100., "f_score": 0.}
    x,y,z,scale = np.load(groundtruth_npz)["normalization_parameters"]
    regressed_trimesh.vertices *= scale
    regressed_trimesh.vertices += np.array([[x[0],y[0],z[0]]])

    iou = intersection_over_union(groundtruth_mesh_file, regressed_trimesh)
    chamfer_l1, chamfer_l2, f_score = chamfer_distance(groundtruth_mesh_file, regressed_trimesh)

    return {"iou": iou.item(), "chamfer_l1": chamfer_l1.item(), "chamfer_l2": chamfer_l2.item(), "f_score": f_score.item()}


def evaluate_patch_network_metrics():

    results_folder = "root_folder/results/experiment_1/"

    faust = False
    if faust:
        evaluate_json = "code/examples/splits/dfaust_20th_test.json"
        data_source = "faust/preprocessed/SdfSamples/"
        dataset_name = "Faust"
        groundtruth_meshes_folder = "faust/obj/"
    else:
        evaluate_json = "code/examples/splits/sv1_joined0_test.json"
        data_source = "shapenetv1/preprocessed/SdfSamples/"
        dataset_name = "ShapeNetV1"
        groundtruth_meshes_folder = "shapenetv1/watertight/"
        shapenet_categories = "code/shapenet_1000_shortnames.json"
    
    grid_resolution = 128
    num_processes = 16
    generate_meshes = True # first run with True, then run again with False.
    # if False, need to do these things:
    # need to "source activate occnet" (conda environment that is defined on the github page of occupancy networks)
    # need to run from occupancy-network folder (due to imports of im2mesh)

    regressed_meshes_folder = "meshes_" + str(grid_resolution) + "/"
    evaluation_folder = results_folder + "output/evaluation_" + str(grid_resolution) + "/"

    print(evaluation_folder, flush=True)
    create_folder(evaluation_folder)
    
    # generate meshes from latents
    if generate_meshes:
        visualize_mixture(results_folder=results_folder, grid_resolution=grid_resolution, mesh_files=evaluate_json, data_source=data_source, break_if_latent_does_not_exist=True, output_name=regressed_meshes_folder)
        return

    regressed_meshes_folder = results_folder + "output/parts/" + regressed_meshes_folder

    # evaluate

    import json
    with open(evaluate_json, "r") as evaluate_json:
        evaluate_json = json.load(evaluate_json)
    evaluate_json = evaluate_json[dataset_name]

    import trimesh
    evaluation_results = {}
    for class_, objects_ in evaluate_json.items():

        print("starting with " + class_ + ", containing " + str(len(objects_)) + " objects", flush=True)

        args = [(object_, class_, regressed_meshes_folder, groundtruth_meshes_folder, dataset_name, data_source) for object_ in objects_]
        with Pool(processes=num_processes) as pool:
            listed_results = pool.map(_evaluate_object, args)

        print("finished with " + class_, flush=True)

        evaluation_results[class_] = dict(zip(objects_, listed_results))
            
    # write out quantitative results

    import json
    with open(evaluation_folder + "individual_results.json", "w") as output_json:
        json.dump(evaluation_results, output_json, indent=2)

    average_results = {}
    for class_ in list(evaluation_results.keys()):
        average_results[class_] = {}

        ious = [ object_["iou"] for object_ in evaluation_results[class_].values() ]
        average_results[class_]["iou"] = np.mean(ious)

        chamfer_l1s = [ object_["chamfer_l1"] for object_ in evaluation_results[class_].values() ]
        average_results[class_]["chamfer_l1"] = np.mean(chamfer_l1s)

        chamfer_l2s = [ object_["chamfer_l2"] for object_ in evaluation_results[class_].values() ]
        average_results[class_]["chamfer_l2"] = np.mean(chamfer_l2s)

        f_scores = [ object_["f_score"] for object_ in evaluation_results[class_].values() ]
        average_results[class_]["f_score"] = np.mean(f_scores)

    print(average_results, flush=True)
    with open(evaluation_folder + "average_results.json", "w") as output_json:
        json.dump(average_results, output_json, indent=2)

    if not faust:

        with open(shapenet_categories, "r") as shapenet_categories:
            shapenet_names = json.load(shapenet_categories)

        shapenet_names = [(cat["id"], cat["name"]) for cat in shapenet_names.values()]
        shapenet_names = sorted(shapenet_names, key=lambda x: x[1])
        
        with open(evaluation_folder + "average_results_shapenet.txt", "w") as output_txt:
            for id, name in shapenet_names:
                line = name + " " + "{:.1f}".format(average_results[id]["iou"]) + " " + "{:.3f}".format(average_results[id]["chamfer_l2"]) + " " + "{:.1f}".format(average_results[id]["f_score"]) + "\n"
                output_txt.write(line)
            
            mean_iou = np.mean([average_results[id]["iou"] for id, _ in shapenet_names])
            mean_chamfer_l2 = np.mean([average_results[id]["chamfer_l2"] for id, _ in shapenet_names])
            mean_f_score = np.mean([average_results[id]["f_score"] for id, _ in shapenet_names])
            line = "mean " + "{:.1f}".format(mean_iou) + " " + "{:.3f}".format(mean_chamfer_l2) + " " + "{:.1f}".format(mean_f_score) + "\n"
            output_txt.write(line)


def generate_watertight_meshes_and_sample_points():

    folders_to_process = ["ShapeNetCore.v1/04256520/",
                          "ShapeNetCore.v1/02691156/",
                          "ShapeNetCore.v1/03636649/",
                          "ShapeNetCore.v1/04401088/",
                          "ShapeNetCore.v1/04530566/",
                          "ShapeNetCore.v1/03691459/",
                          "ShapeNetCore.v1/03001627/",
                          "ShapeNetCore.v1/02933112/",
                          "ShapeNetCore.v1/04379243/",
                          "ShapeNetCore.v1/03211117/",
                          "ShapeNetCore.v1/02958343/",
                          "ShapeNetCore.v1/02828884/",
                          "ShapeNetCore.v1/04090263/"] # list of folders that contain objs
    output_folder = "shapenetv1/watertight/"
    train_val_test_split = [0.75, 0.8]
    joined_jsons = [[0,1,2,3,4,5,6,7,8,9,10,11,12]] # each sublist indexes "folders_to_process"
    generate_command = True # first run with True, then run again with False

    json_prefix = "sv1_"
    num_processes = 8

    dataset_name = "ShapeNetV1"
    mesh_fusion_path = "occupancy_networks-master/external/mesh-fusion/"
    sample_mesh_path = "occupancy_networks-master/sample_mesh.py"
    json_folder = "code/examples/splits/"

    command_file = output_folder + "command.sh"

    create_folder(output_folder)

    json_dicts = []
    command = ""
    for i, class_input_folder in enumerate(folders_to_process):
        # input_class_folder: contains .objs

        class_name = os.path.split(class_input_folder[:-1])[1]
        class_output_folder = output_folder + class_name + "/"

        intermediate_scaled_folder = class_output_folder + "scaled/"
        intermediate_transform_folder = class_output_folder + "transform/"
        intermediate_depth_folder = class_output_folder + "depth/"
        intermediate_watertight_folder = class_output_folder + "intermediate_watertight/"

        create_folder(class_output_folder)
        create_folder(intermediate_scaled_folder)
        create_folder(intermediate_transform_folder)
        create_folder(intermediate_depth_folder)
        create_folder(intermediate_watertight_folder)

        if generate_command:

            # scale meshes
            command += "python " + mesh_fusion_path + "1_scale.py --n_proc " + str(num_processes) + " --in_dir " + class_input_folder + " --out_dir " + intermediate_scaled_folder + " --t_dir " + intermediate_transform_folder + "\n"
            # depth maps
            command += "python " + mesh_fusion_path + "2_fusion.py --mode=render --n_proc " + str(num_processes) + " --in_dir " + intermediate_scaled_folder + " --out_dir " + intermediate_depth_folder + "\n"
            ## make watertight
            command += "python " + mesh_fusion_path + "2_fusion.py --mode=fuse --n_proc " + str(num_processes) + " --in_dir " + intermediate_depth_folder + " --out_dir " + intermediate_watertight_folder + " --t_dir " + intermediate_transform_folder + "\n"
            # do some weird final scaling for OccNet-compatibility
            command += "python " + sample_mesh_path + " " + intermediate_watertight_folder + " --n_proc " + str(num_processes) + " --resize --bbox_in_folder " + class_input_folder + " --mesh_folder " + class_output_folder + "\n"
        else:
            # create json
            meshes = os.listdir(class_output_folder)
            meshes = sorted([file for file in meshes if file[-4:] == ".obj"])
            num_meshes = len(meshes)
            train_meshes = meshes[0 : int(train_val_test_split[0] * num_meshes)]
            val_meshes = meshes[int(train_val_test_split[0] * num_meshes) : int(train_val_test_split[1] * num_meshes)]
            test_meshes = meshes[int(train_val_test_split[1] * num_meshes) : num_meshes]
        
            def _create_json(mesh_list, json_name):
                if mesh_list:
                    dict = {dataset_name: {class_name: mesh_list}}
                    import json
                    json_file = json_folder + json_prefix + json_name
                    with open(json_file, "w") as json_file:
                        json.dump(dict, json_file, indent=2)
            _create_json(train_meshes, class_name + "_train.json")
            _create_json(val_meshes, class_name + "_val.json")
            _create_json(test_meshes, class_name + "_test.json")
            json_dicts.append( {"train": train_meshes, "val": val_meshes, "test": test_meshes, "class": class_name} )

    if generate_command:
        with open(command_file, "w") as command_file:
            command_file.write(command)
    else:
        for i, joined_json in enumerate(joined_jsons):
            train_dict = {}
            val_dict = {}
            test_dict = {}
            for _class in joined_json:
                class_dict = json_dicts[_class]
                class_name = class_dict["class"]
                train_dict[class_name] = class_dict["train"]
                val_dict[class_name] = class_dict["val"]
                test_dict[class_name] = class_dict["test"]
            import json
            json_file = json_folder + json_prefix + "joined" + str(i) + "_train.json"
            with open(json_file, "w") as json_file:
                json.dump({dataset_name: train_dict}, json_file, indent=2)
            json_file = json_folder + json_prefix + "joined" + str(i) + "_val.json"
            with open(json_file, "w") as json_file:
                json.dump({dataset_name: val_dict}, json_file, indent=2)
            json_file = json_folder + json_prefix + "joined" + str(i) + "_test.json"
            with open(json_file, "w") as json_file:
                json.dump({dataset_name: test_dict}, json_file, indent=2)  



def hierarchical_representation():
    # untested code, never used
    
    import torch
    
    results_folder = ""
    sdf_samples = ".npz"
    output_name = "mpu/"

    patches_per_patch_area = 8
    max_num_patches = 200
    hierarchy_factor = 2 # factor by which the patch radius is downscaled per hierarchy level
    surface_sdf_threshold = 0.02

    initial_learning_rate = 0.001
    num_iterations = 1000
    do_code_regularization = True
    sdf_samples_per_forward = 8000

    output_folder = results_folder + "output/" + output_name
    create_folder(output_folder)
    backupThisFile(output_folder)

    device = torch.cuda.current_device()

    # network
    network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs = _setup_torch_network(results_folder)
    default_patch_size = network.patch_clamping_radius
    sdf_clamping_distance = specs["ClampingDistance"]

    # SDF samples
    sdf_samples_np = np.load(sdf_samples)
    def _process_tensor(tensor):
        tensor_nan = np.isnan(tensor[:,3])
        tensor = tensor[~tensor_nan, :]
        tensor[:,3] = np.clip(tensor[:,3], -sdf_clamping_distance, +sdf_clamping_distance)
        return tensor
    def process_tensor_2(pos_neg):
        return _process_tensor(np.concatenate([sdf_samples_np[pos_neg], sdf_samples_np[pos_neg + "_normals"]], axis=0))
    sdf_samples_split_np = (process_tensor_2("pos"), process_tensor_2("neg"))
    sdf_samples_np = np.concatenate((sdf_samples_split_np[0], sdf_samples_split_np[1]), axis=0) # x,y,z, SDF, nx,ny,nz
    sdf_samples_torch = (torch.from_numpy(sdf_samples_split_np[0][:,:4]).cuda(), torch.from_numpy(sdf_samples_split_np[1][:,:4]).cuda()) # pos & SDF, no normals
    sdf_samples_torch[0].requires_grad = False
    sdf_samples_torch[1].requires_grad = False

    def unpack_sdf_samples_from_ram(sdf_samples, masks, subsample=None):
        if subsample is None:
            return sdf_samples
        pos_tensor = sdf_samples[0]
        neg_tensor = sdf_samples[1]
        pos_size = pos_tensor.shape[0]
        neg_size = neg_tensor.shape[0]
        pos_mask, neg_mask = masks

        # split the sample into half
        half = int(subsample / 2)

        if pos_size <= half:
            random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
            sample_pos = torch.index_select(pos_tensor, 0, random_pos)
            sample_pos_mask = torch.index_select(pos_mask, 0, random_pos)
        else:
            pos_start_ind = random.randint(0, pos_size - half)
            sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]
            sample_pos_mask = pos_mask[pos_start_ind : (pos_start_ind + half)]

        if neg_size <= half:
            random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
            sample_neg = torch.index_select(neg_tensor, 0, random_neg)
            sample_neg_mask = torch.index_select(neg_mask, 0, random_neg)
        else:
            neg_start_ind = random.randint(0, neg_size - half)
            sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]
            sample_neg_mask = neg_mask[neg_start_ind : (neg_start_ind + half)]

        samples = torch.cat([sample_pos, sample_neg], 0)
        samples_mask = torch.cat([sample_pos_mask, sample_neg_mask], 0)
        return samples, samples_mask

    # learning rate schedule
    def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_group:
            param_group["lr"] = lr
    decreased_by = 2
    adjust_lr_every = max(int(num_iterations / 5), 1)

    # initialize list of patches with giant patch area (patch scale = sqrt(3) for normalized input in [-1,+1]^3)
    # ( "patch"/"patch area", latent, patch center, patch rotation, patch scale, hierarchy level)
    initial_full_patch_radius = np.sqrt(3) / default_patch_size
    patch_list = [{"type": "patch area", "latent": None, "position": torch.zeros(3), "rotation": torch.zeros(3), "scale": initial_full_patch_radius * torch.ones(1), "level": 1, "patch_error": np.inf}]

    # keep looping until all patch errors below threshold (or max num patches is reached)
    while True:

        # pick a patch area from list of patches
        candidates = [ (i, patch["patch_error"]) for i, patch in enumerate(patch_list) if x[0] == "patch area" ]
        if candidate_indices:
            highest_patch_error_index = np.argmax( np.array([ error for i, error in candidates ]))
            current_patch_area = patch_list.pop(candidates[highest_patch_error_index][0])
        else:
            # no more patch_areas exist
            break

        # setup patch area information
        initial_patch_radius = initial_full_patch_radius / (hierarchy_factor ** current_patch_area["level"])
        # all patch scales are possible, put each hierarchy level has it own, disjoint range of possible patch scales. mathematically: "u*x = (a*x) / u" with a = hierarchy_factor --> u = sqrt(a)
        max_patch_radius = initial_patch_radius * np.sqrt(hierarchy_factor)
        min_patch_radius = initial_patch_radius / np.sqrt(hierarchy_factor)

        # get SDF samples from this area (with additional boundary samples such that patches can move to the border of the patch area and still have samples outside the patch area)
        sdf_positions = sdf_samples_np[:,:3]
        distances = sdf_positions - current_patch_area["position"].unsqueeze(0).detach().cpu().numpy()
        distances = np.linalg.norm(distances, axis=1) # shape: samples
        inside_mask = distances <= current_patch_area["scale"]
        boundary_mask = ~inside_mask & (distances <= current_patch_area["scale"] + max_patch_radius)
        #outside_mask = ~inside_mask & ~boundary_mask
        #sdf_samples_patch_area_np = sdf_samples_np[inside_mask | boundary_mask,:]

        # get surface SDF samples from this area
        surface_mask = np.abs(sdf_samples_np[:,3]) <= surface_sdf_threshold

        # farthest point sampling of surface SDF samples to initialize positions
        farthest_points_positions, farthest_points_mask, farthest_points_indices = farthest_point_sampling(sdf_samples_np[surface_mask & inside_mask,:3], patches_per_patch_area)
        if farthest_points_mask is None: # i.e. if not enough surface points exist --> some points appear multiple times --> add noise to distinguish them
            farthest_points_positions += np.random.normal(scale=0.05 * current_patch_area["scale"], size=farthest_points_positions.shape)
        
        # patch rotations
        farthest_points_rotations = np.array([_get_rotation_matrix_from_normal(normal) for normal in sdf_samples_np[surface_mask & inside_mask,4:][farthest_points_indices] ]) # num_patches x 3

        ## reconstruct SDF samples from this area

        # initialize subpatches of this patch area
        network.num_patches = patches_per_patch_area
        subpatch_list = []
        for position, rotation in zip(farthest_points_positions, farthest_points_rotations):
            patch = { "type": "patch", "latent": torch.zeros(patch_latent_size).cuda(), "position": torch.from_numpy(position).cuda(), "rotation": torch.from_numpy(rotation).cuda(), "scale": torch.from_numpy(initial_patch_radius).cuda(), "level": current_patch_area["level"] + 1 }

            subpatch_list.append(patch)

        # initialize optimizer
        optimizer = torch.optim.Adam({ "params": [patch["latent"] for patch in subpatch_list] + [patch["position"] for patch in subpatch_list] + [patch["rotation"] for patch in subpatch_list] + [patch["scale"] for patch in subpatch_list], "lr": initial_learning_rate})

        # convert to Pytorch and send to CUDA
        def split_mask(mask):
            pos_size = sdf_samples_split_np[0].shape[0]
            pos_mask = mask[:pos_size]
            neg_mask = mask[pos_size:]
            return pos_mask, neg_mask
        pos_inside_mask, neg_inside_mask = split_mask(inside_mask)
        pos_boundary_mask, neg_boundary_mask = split_mask(boundary_mask)

        # pos/neg masks from "all" to "inside|boundary"
        pos_subsample_mask = pos_inside_mask | pos_boundary_mask
        pos_inside_mask = pos_inside_mask[pos_subsample_mask]

        neg_subsample_mask = neg_inside_mask | neg_boundary_mask
        neg_inside_mask = neg_inside_mask[neg_subsample_mask]

        # to CUDA
        pos_subsample_mask_torch = torch.from_numpy(pos_subsample_mask).cuda()
        pos_inside_mask_torch = torch.from_numpy(pos_inside_mask).cuda()
        neg_subsample_mask_torch = torch.from_numpy(neg_subsample_mask).cuda()
        neg_inside_mask_torch = torch.from_numpy(neg_inside_mask).cuda()

        # SDF samples
        sdf_samples_patch_area_split_torch = (sdf_samples_torch[0][pos_subsample_mask_torch,:], sdf_samples_torch[1][neg_subsample_mask_torch,:])

        for current_iteration in range(num_iterations):
            # learning rate
            adjust_learning_rate(initial_learning_rate, optimizer, current_iteration, decreased_by, adjust_lr_every)

            # zero gradient
            optimizer.zero_grad()

            # subsample SDF samples and masks to "sdf_samples_per_forward" many samples
            current_sdf_samples, current_inside_masks = unpack_sdf_samples_from_ram(sdf_samples_patch_area_split_torch, (pos_inside_mask_torch, neg_inside_mask_torch), subsample=sdf_samples_per_forward)

            # build mixture latent vector
            mixture_latent_vector = torch.cat([ torch.cat([patch["latent"], patch["position"], patch["rotation"], patch["scale"]], 0) for patch in subpatch_list], 0)

            # inputs dict
            inputs = {}
            inputs["mixture_latent_vectors"] = mixture_latent_vector.unsqueeze(0)
            inputs["xyz"] = current_sdf_samples[:,:3].reshape(1, -1, 3)
            inputs["sdf_gt"] = current_sdf_samples[:,3].reshape(1, -1, 1)
            inputs["use_encoder"] = False
            inputs["num_samp_per_scene"] = sdf_samples_per_forward
            inputs["extra_losses_mask"] = current_inside_masks

            # forward
            pred_sdf, extra_losses = network(inputs)

            # compute loss
            loss = torch.zeros(1, device=device)[0]
            for extra_loss_name, extra_loss_value in extra_losses.items():
                if "latent_regularization" in extra_loss_name:
                    loss += 1e-4 * extra_loss_value
                else:
                    loss += extra_loss_value

            # backward
            loss.backward()
            optimizer.step()

            # clip patch position inside patch area
            print("this might not modify the tensor in-place and instead create a new tensor which is then no longer optimized?")
            patch_positions = torch.stack([patch["position"] for patch in subpatch_list], dim=0) # num_patches x 3
            patch_distances = pach_positions - current_patch_area["position"].unsqueeze(0)
            patch_distances = torch.norm(patch_positions, dim=1)
            patches_too_far_away_mask = patch_distances > current_patch_area["scale"]
            patch_positions[patches_too_far_away_mask,:] = torch.nn.functional.normalize( patch_distances[patches_too_far_away_mask,:], p=2, dim=1 ) * current_patch_area["scale"] + current_patch_area["position"].unsqueeze(0)
            patch_positions = patch_positions.detach()
            for patch, new_patch_position in zip(subpatch_list, patch_positions):
                patch["position"] = new_patch_position

            # clip patch scale to 1 +- (1/3)
            for patch in subpatch_list:
                patch["scale"] = torch.clamp(patch["scale"], min_patch_radius, max_patch_radius).detach()

        # get per-patch reconstruction errors
        per_patch_errors = network.direct_patch_loss.detach().cpu().numpy()[0] # shape: num_patches ([0] takes the first scene since there is only one)
        for patch_error, subpatch in zip(per_patch_errors, subpatch_list):
            subpatch["patch_error"] = patch_error

        # insert patches into patch list
        subpatch_list.sort(key=lambda x: x["patch_error"], reverse=True) # first consider potential patch areas with the worst error (since we might not have enough max_num_patches available to turn all patches into patch areas)
        for i, patch in enumerate(subpatch_list):

            # further divide patch under certain conditions
            error_condition = patch["patch_error"] > patch_error_threshold
            # only subdivide patch if existing patch_list, subpatch_list and the to-be-created subpatches all together are fewer than max_num_patches
            num_patches_condition = len([_ for patch in patch_list if patch[type] == "patch"])\
                + len([_ for patch in patch_list if patch[type] == "patch area"]) * patches_per_patch_area\
                + (len(subpatch_list) - i - 1) \
                + patches_per_patch_area <= max_num_patches

            if error_condition and num_patches_condition:
                # convert patch to patch area
                patch["type"] = "patch area"
            patch["latent"] = patch["latent"].detach()
            patch["position"] = patch["position"].detach()
            patch["rotation"] = patch["rotation"].detach()
            patch["scale"] = patch["scale"].detach()
            patch_list.append(patch) 


def interpolate_deepsdf(results_folder=None, grid_resolution=None):
    import torch

    if results_folder is None:
        results_folder = "/"
    checkpoint = "latest" #None

    to_interpolate = [(6,1), (6,33), (7,1), (34, 38), (34, 42), (17,6), (34,33)]
    
    interpolate_patches = False

    if grid_resolution is None:
        grid_resolution = 128
    output_name = "parts_interpolate1"
    mesh_files = None
    data_source = None

    if checkpoint is None:
        checkpoint = "latest"
    output_name += "_" + checkpoint + "_" + str(grid_resolution)
    output_name += "/"
    output_folder = results_folder + "output/parts/" + output_name

    create_folder(output_folder)

    network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs = _setup_torch_network(results_folder)
    load_weights_into_network(network, checkpoint=checkpoint)
    network.use_depth_encoder = False

    with open(results_folder + "backup/special_files/npyfilenames.txt", "r") as train_split:
        train_split = train_split.readlines()#[:-1]
        train_split = [line[:-5] for line in train_split] # remove ".npz\n"

    if mesh_files is None:
        mesh_files = train_split
    else:
        import json
        if mesh_files[-5:] == ".json":
            with open(mesh_files, "r") as mesh_files:
                mesh_files = list(json.load(mesh_files).items())
            for super_folder, entries in mesh_files:
                for subfolder, subentries in entries.items():
                    mesh_files = [super_folder + "/" + subfolder + "/" + mesh_file for mesh_file in subentries]
        else:
            with open(mesh_files, "r") as mesh_files:
                mesh_files = mesh_files.readlines()[:-1]
        with open(results_folder + "backup/specs.json", "r") as specs:
            specs = "\n".join([line for line in specs.readlines() if line.strip()[:2] != "//"]) # remove comment lines
            specs = json.loads(specs)
        if data_source is None:
            data_source = specs["DataSource"] + "SdfSamples/"

    with open(output_folder + "MeshFiles", "w") as output_mesh_files:
        for mesh_file in mesh_files:
            output_mesh_files.write(mesh_file + ".obj\n")

    # determine latent codes of known meshes
    train_latent_codes = torch.load(results_folder + "LatentCodes/" + checkpoint + ".pth")
    mesh_files_latent_codes = []
    for i, mesh in enumerate(mesh_files):
        if "____rejected.npz" in mesh:
            mesh = mesh[:-len("____rejected.npz")]
        train_id = train_split.index(mesh)
        latent_code = train_latent_codes["latent_codes"][train_id,0,:].unsqueeze(0).clone().cuda()
        mesh_files_latent_codes.append((mesh, latent_code))

    # generate interpolated meshes
    for a, b in to_interpolate:
        interpolation_steps = 60
        for i in range(interpolation_steps):
            latent_a = mesh_files_latent_codes[a][1]
            latent_b = mesh_files_latent_codes[b][1]
            
            if interpolate_patches: # only use first patch
                latent_a = latent_a[:patch_latent_size]
                latent_b = latent_b[:patch_latent_size]
            
            alpha = i/float(interpolation_steps-1)
            print(alpha)
            latent_code = alpha * latent_b + (1.-alpha) * latent_a
            vertices, faces = latent_to_mesh(latent_code, grid_resolution=grid_resolution, patch_only=interpolate_patches)
            mesh_folder = os.path.split(output_folder + mesh)[0]
            create_folder(mesh_folder)
            instance_name =  mesh + "_" + str(a).zfill(5) + "_" + str(b).zfill(5) + "_" + str(i).zfill(4)
            _write_mesh(vertices, faces+1, output_folder + instance_name + ".obj")
            torch.save( {"epoch": None, "latent_codes": latent_code.reshape(1,1,-1)}, output_folder + instance_name + "_latent.pth")
            
            

def generative_model_deepsdf(results_folder=None, grid_resolution=None):
    import torch

    if results_folder is None:
        results_folder = "experiment_1/"
    checkpoint = "latest" #None

    num_random_latent_samples_for_generative_model = 50

    if grid_resolution is None:
        grid_resolution = 128
    output_name = "parts_generate_train/"

    output_folder = results_folder + "output/parts/" + output_name

    create_folder(output_folder)

    network, patch_latent_size, mixture_latent_size, load_weights_into_network, sdf_to_latent, patch_forward, latent_to_mesh, get_training_latent_vectors, specs = _setup_torch_network(results_folder)
    load_weights_into_network(network, checkpoint=checkpoint)

    # determine latent codes of known meshes
    train_latent_codes = torch.load(results_folder + "LatentCodes/" + checkpoint + ".pth")
    if num_random_latent_samples_for_generative_model > 0:
        known_latent_codes = train_latent_codes["latent_codes"]
        known_latent_codes = known_latent_codes.squeeze(1)

        means = torch.mean(known_latent_codes, dim=0).detach().numpy() # shape: latent_size
        standard_deviations = torch.std(known_latent_codes, dim=0).detach().numpy() # shape: latent_size

        covariance = np.cov(m=known_latent_codes.detach(), bias=True, rowvar=False)

    for i in range(num_random_latent_samples_for_generative_model):

        #latent_code = np.random.normal(means, standard_deviations, size=known_latent_codes.shape[-1])
        latent_code = np.random.multivariate_normal(means, covariance)

        latent_code = torch.from_numpy(latent_code)
        vertices, faces = latent_to_mesh(latent_code.reshape(1,-1).float().clone().cuda(), grid_resolution=grid_resolution)
        _write_mesh(vertices, faces+1, output_folder + "generative_" + str(i) + ".obj")
        torch.save( {"epoch": None, "latent_codes": latent_code.reshape(1,1,-1)}, output_folder + "generative_" + str(i) + "_latent.pth")


def main():
    ## Evaluation
    evaluate_patch_network_metrics()
    
    ## Making ShapeNet watertight
    #generate_watertight_meshes_and_sample_points()
    
    ## Converting latent codes into meshes
    #visualize_parts_individually # patches
    #visualize_mixture() # full object

    ## Applications
    #interpolate_deepsdf()
    #generative_model_deepsdf()
    #shape_completion_deepsdf()
    #hierarchical_representation()

if __name__ == "__main__":
    main()