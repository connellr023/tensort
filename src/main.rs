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
use tch::vision::resnet;
use crate::models::arguments_model::ArgumentsModel;
use crate::models::cnn_model::CNNModel;
use crate::views::results_view::*;
use crate::controllers::io_controller::*;
use crate::controllers::embeddings_controller::*;

const VARSTORE_BYTES: &[u8] = include_bytes!("../resnet34.ot");

fn run(args: Vec<String>) -> Result<(), Box<dyn Error>>
{
	// Read in command line arguments
	let args = ArgumentsModel::from(args)?;

	// Print selected arguments
	println!("{}", args);

	// Initialize convolutional nesural network and print related info
	let model = CNNModel::new(VARSTORE_BYTES, resnet::resnet34)?;
	println!("{}\n", model);

	// Read the target dir and process each image
	println!("Generating image embeddings...");
	let (embeddings, image_paths, missed_image_paths) = gen_image_embeddings(args.target_dir(), &model)?;

	// If some images failed to process, list them
	if missed_image_paths.len() > 0 {
		println!("{}", format_missed_images(missed_image_paths));
	}

	// Group embeddings together
	println!("Computing similarities and clustering embeddings...");
	let similarities = calc_pairwise_cosine_similarities(embeddings.as_slice());
	let similarity_table = cluster_embeddings(similarities.as_slice(), embeddings.len(), args.class_count());

	// Generate class names if option is set
	let class_names = if args.should_not_gen_names() {
		gen_default_class_names(args.class_count())
	}
	else {
		println!("Averaging tensors and deriving class names...");
		gen_class_names(embeddings.as_slice(), &similarity_table)
	};

	// Manipulate file locations
	println!("Moving files...");
	update_target_dir(args.target_dir(), image_paths.as_slice(), class_names.as_slice(), &similarity_table)?;

	// Print classification results
	// Move all the values since this is the end
	print!("\nResults:\n{}", format_classified_images(similarity_table, image_paths, class_names));

	Ok(())
}

fn main()
{
	let args = args().collect();

	if let Err(err) = run(args) {
		eprintln!("{}", err);
	}
}