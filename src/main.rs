// Tensor-Sort Command Line Tool
// Author: Connell Reffo
// Developed: 2024
#![crate_name = "tensort"]

mod models;
mod views;
mod controllers;
mod errors;

use std::env::args;
use tch::Tensor;
use tch::vision::resnet::resnet34;
use crate::models::arguments_model::ArgumentsModel;
use crate::models::cnn_model::CNNModel;
use crate::controllers::embeddings_controller::*;

const PRETRAINED_MODEL_PATH: &str = "/home/connell/Programming/model-stuff/resnet34.ot";

fn main()
{
	// Read in command line arguments
	let args = match ArgumentsModel::new(args().collect()) {
		Ok(args) => args,
		Err(err) => {
			eprintln!("{}", err);
			return;
		}
	};

	// Print selected arguments
	println!("{}", args);

	// Initialize convolutional neural network and print related info
	let model = match CNNModel::new(PRETRAINED_MODEL_PATH, resnet34) {
		Ok(model) => model,
		Err(err) => {
			eprintln!("{}", err);
			return;
		}
	};

	println!("{}", model);

	// Read the target dir and process each image
	let (embeddings, missed_images) = match gen_image_embeddings(args.target_dir(), &model) {
		Ok(result) => result,
		Err(err) => {
			eprintln!("{}", err);
			return;
		}
	};

	for embedding in embeddings.iter()
	{
		println!("path: {}", embedding.1.to_str().unwrap());
	}

	let binding = embeddings
		.iter()
		.map(|tuple| { &tuple.0 })
		.collect::<Vec<&Tensor>>();

	let embeddings_slice = binding
			.as_slice();

	let similarities = calc_pairwise_cosine_similarities(embeddings_slice);

	// for similarity in similarities.iter()
	// {
	// 	println!("similarity: {}\n", similarity);
	// }

	// println!("threshold: {}\n", embedding_controller::calc_similarity_threshold(similarities.as_slice(), 6));
	let similarity_table = cluster_embeddings(similarities.as_slice(), embeddings.len(), args.class_count());

	for i in 0..similarity_table.len() {
		println!("Class {}:", i);

		for j in 0..similarity_table[i].len() {
			println!("class path: {}", embeddings[similarity_table[i][j]].1.to_str().unwrap());
		}

		println!();
	}
}