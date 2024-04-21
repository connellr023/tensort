use tch::{Device, TchError, Tensor};
use tch::nn::{FuncT, VarStore};
use tch::vision::{imagenet, resnet};

pub struct ResNet18Model
{
    device: Device,
    varstore: VarStore,
    model: FuncT<'static>
}

impl ResNet18Model
{
    pub fn new(pretrained_path: &'static str) -> Result<Self, TchError>
    {
        let device = Device::cuda_if_available();
        let mut varstore = VarStore::new(device);
        let model = resnet::resnet18(&varstore.root(), imagenet::CLASS_COUNT);

        varstore.load(pretrained_path)?;
        
        Ok(Self {
            device,
            varstore,
            model
        })
    }

    pub fn device(&self) -> Device
    {
        self.device
    }

    pub fn gen_embedding(&self, image_path: &str, activation_func: fn(Tensor) -> Tensor) -> Result<Tensor, TchError>
    {
        let image = imagenet::load_image(image_path)?;
        
        let image = image.to_device(self.varstore.device())
            .unsqueeze(0)
            .apply_t(&self.model, false);

        Ok(activation_func(image))
    }
}