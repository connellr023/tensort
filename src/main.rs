mod models;
mod views;
mod controllers;
mod errors;

use std::error::Error;
use std::env::args;
use tch::Tensor;

use crate::models::arguments_model::ArgumentsModel;
use crate::models::resnet18_model::ResNet18Model;
use crate::controllers::embedding_controller;
// use tch::Kind;
// use tch::vision::imagenet::top;

// const IMAGE_PATH: &str = "/home/connell/Programming/model-stuff/images/barbet.jpg";
const PRETRAINED_MODEL_PATH: &str = "/home/connell/Programming/model-stuff/resnet18.ot";

fn main() -> Result<(), Box<dyn Error>>
{
	let args = ArgumentsModel::new(args().collect())?;
	let dir = args.target_dir();

	println!("{}", dir.to_str().unwrap());

	let model = ResNet18Model::new(PRETRAINED_MODEL_PATH)?;
	let embeddings = embedding_controller::gen_image_embeddings(dir, &model)?;
	print!("Selected device: {:?}\n", model.device());

	for embedding in embeddings.iter()
	{
		println!("path: {}\n", embedding.1.to_str().unwrap());
	}

	let similarities = embedding_controller::calc_pairwise_cosine_similarities(
		embeddings
			.iter()
			.map(|tuple| { &tuple.0 })
			.collect::<Vec<&Tensor>>()
			.as_slice()
	);

	for similarity in similarities.iter()
	{
		println!("similarity: {}\n", similarity);
	}

	println!("threshold: {}\n", embedding_controller::calc_similarity_threshold(similarities.as_slice(), 6));

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