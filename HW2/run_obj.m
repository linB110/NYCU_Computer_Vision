load('sfm_data.mat');

if ~exist('output_dir', 'dir')
    mkdir('output_dir');
end

obj_main(P, p_img2, M, tex_name, 1, 'output_dir');