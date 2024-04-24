// Tensor-Sort Command Line Tool
// Author: Connell Reffo
// Developed: 2024
#![crate_name = "tensort"]

mod models;
mod views;
mod controllers;
mod errors;

use std::error::Error;
use std::env::args;
use tch::vision::resnet::resnet34;
use crate::models::arguments_model::ArgumentsModel;
use crate::models::cnn_model::CNNModel;
use crate::views::results_view::*;
use crate::controllers::embeddings_controller::*;

const PRETRAINED_MODEL_PATH: &str = "/home/connell/Programming/model-stuff/resnet34.ot";

fn run(args: Vec<String>) -> Result<(), Box<dyn Error>>
{
	// Read in command line arguments
	let args = ArgumentsModel::new(args)?;

	// Print selected arguments
	println!("{}", args);

	// Initialize convolutional neural network and print related info
	let model = CNNModel::new(PRETRAINED_MODEL_PATH, resnet34)?;
	println!("{}", model);

	// Read the target dir and process each image
	println!("Generating image embeddings...");
	let (embeddings, missed_images) = gen_image_embeddings(args.target_dir(), &model)?;

	// If some images failed to process, list them
	if missed_images.len() > 0 {
		println!("{}", format_missed_images(missed_images));
	}

	// Group embeddings together
	println!("Computing similarities and clustering embeddings...\n");
	let similarities = calc_pairwise_cosine_similarities(embeddings.as_slice());
	let similarity_table = cluster_embeddings(similarities.as_slice(), embeddings.len(), args.class_count());

	// Generate class names if option is set
	let class_names = if args.should_gen_names() { gen_class_names(embeddings.as_slice(), &similarity_table) } else { vec![] };

	// Print classification results
	print!("{}", format_classified_images(similarity_table, embeddings, class_names));

	Ok(())
}

fn main()
{
	let args = args().collect();

	if let Err(err) = run(args) {
		eprintln!("{}", err);
	}
}