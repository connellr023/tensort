// Tensor-Sort Command Line Tool
// Author: Connell Reffo
// Developed: 2024
#![crate_name = "tensort_cli"]

mod models;
mod views;
mod controllers;
mod errors;

use std::error::Error;
use std::env::args;
use tch::Tensor;
use tch::vision::resnet::resnet34;

use crate::models::arguments_model::ArgumentsModel;
use crate::models::cnn_model::CNNModel;
use crate::controllers::embedding_controller;
// use tch::Kind;
// use tch::vision::imagenet::top;

// const IMAGE_PATH: &str = "/home/connell/Programming/model-stuff/images/barbet.jpg";
const PRETRAINED_MODEL_PATH: &str = "/home/connell/Programming/model-stuff/resnet34.ot";

fn main() -> Result<(), Box<dyn Error>>
{
	let args = ArgumentsModel::new(args().collect())?;
	let dir = args.target_dir();

	println!("{}", dir.to_str().unwrap());

	let model = CNNModel::new(PRETRAINED_MODEL_PATH, resnet34)?;
	let embeddings = embedding_controller::gen_image_embeddings(dir, &model)?;
	println!("Selected device: {:?}", model.device());

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

	let similarities = embedding_controller::calc_pairwise_cosine_similarities(embeddings_slice);

	// for similarity in similarities.iter()
	// {
	// 	println!("similarity: {}\n", similarity);
	// }

	// println!("threshold: {}\n", embedding_controller::calc_similarity_threshold(similarities.as_slice(), 6));
	let similarity_table = embedding_controller::cluster_embeddings(similarities.as_slice(), embeddings.len(), 4);

	for i in 0..similarity_table.len() {
		println!("Class {}:", i);

		for j in 0..similarity_table[i].len() {
			println!("class path: {}", embeddings[similarity_table[i][j]].1.to_str().unwrap());
		}

		println!();
	}

	// let embedding = model_handler.gen_embedding(IMAGE_PATH, |tensor| {
	// 	tensor.softmax(-1, Kind::Float)
	// })?;

	// println!("Tensor embedding of the image:\n{}", embedding);
	
	// for (probability, class) in top(&embedding, 5).iter()
	// {
	// 	println!("{:50} {:5.2}%", class, probability * 100.0);
	// }

	Ok(())
}