use tch::{Device, Kind};
use tch::nn::VarStore;
use tch::vision::{
	imagenet,
	resnet
};

const IMAGE_PATH: &str = "/home/connell/Programming/model-stuff/images/barbet.jpg";
const PRETRAINED_MODEL_PATH: &str = "/home/connell/Programming/model-stuff/resnet18.ot";

fn main() -> Result<(), Box<dyn std::error::Error>>
{
	// let descriptions = vec![
	// 	"man",
	// 	"dog",
	// 	"bodybuilder",
	// 	"bug",
	// 	"ocean",
	// 	"weird"
	// ];

	let device = Device::cuda_if_available();
	let mut vs = VarStore::new(device);
	print!("Device selected: {:?}\n", device);

	let image = imagenet::load_image(IMAGE_PATH)?.to_device(vs.device());
	let model = resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);

	vs.load(PRETRAINED_MODEL_PATH)?;

	let output = image
		.unsqueeze(0)
		.apply_t(&model, false)
		.softmax(-1, Kind::Float);

	println!("Top tensor embedding the image:\n{}", output);
	
	for (probability, class) in imagenet::top(&output, 5).iter()
	{
		println!("{:50} {:5.2}%", class, probability * 100.0);
	}

	Ok(())
}