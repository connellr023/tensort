use tch::{Device, Tensor};
use tch::nn::{FuncT, VarStore};
use tch::vision::{imagenet, resnet};

pub struct ResNet18Handler
{
    device: Device,
    varstore: VarStore,
    model: FuncT<'static>
}

impl ResNet18Handler
{
    pub fn new(pretrained_path: &'static str) -> Result<Self, &'static str>
    {
        let device = Device::cuda_if_available();
        let mut varstore = VarStore::new(device);
        let model = resnet::resnet18(&varstore.root(), imagenet::CLASS_COUNT);

        match varstore.load(pretrained_path) {
            Ok(_) => {
                Ok(Self {
                    device,
                    varstore,
                    model
                })
            },
            Err(_) => {
                Err("Failed to load pretrained model from path")
            }
        }
    }

    pub fn device(&self) -> Device
    {
        self.device
    }

    pub fn gen_embedding(&self, image_path: &str, activation_func: fn(Tensor) -> Tensor) -> Result<Tensor, &str>
    {
        match imagenet::load_image(image_path)
        {
            Ok(image) => {
                let image = image.to_device(self.varstore.device())
                    .unsqueeze(0)
                    .apply_t(&self.model, false);

                Ok(activation_func(image))
            },
            Err(_) => {
                Err("Failed to load image from path")
            }
        }
    }
}