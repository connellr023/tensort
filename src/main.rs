mod models;
mod views;
mod controllers;
mod errors;

use std::error::Error;
use std::env::args;

use models::arguments_model::ArgumentsModel;
// use tch::Kind;
// use tch::vision::imagenet::top;
// use crate::models::resnet18_model::ResNet18Model;

// const IMAGE_PATH: &str = "/home/connell/Programming/model-stuff/images/barbet.jpg";
// const PRETRAINED_MODEL_PATH: &str = "/home/connell/Programming/model-stuff/resnet18.ot";

fn main() -> Result<(), Box<dyn Error>>
{
	let args = ArgumentsModel::new(args().collect())?;

	println!("{}", args.target_dir());

	// let model_handler = ResNet18Model::new(PRETRAINED_MODEL_PATH)?;
	// print!("Selected device: {:?}\n", model_handler.device());

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