hed_mat_dir='/home/czey/generative_inpainting/training_data/celebahq_c3/validation/edge';
edge_dir='/home/czey/generative_inpainting/training_data/celebahq_c3/validation/edge2';
image_width=256;
threshold=25.0/255.;
small_edge=5;
PostprocessHED(hed_mat_dir,edge_dir,image_width,threshold,small_edge);