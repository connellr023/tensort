use std::path::PathBuf;
use tch::{Device, Kind, TchError, Tensor};
use tch::nn::{Path, FuncT, VarStore};
use tch::vision::imagenet;

pub type CNNSignature = fn(p: &Path, num_classes: i64) -> FuncT<'static>;

pub struct CNNModel
{
    device: Device,
    varstore: VarStore,
    model: FuncT<'static>
}

impl CNNModel
{
    pub fn new(pretrained_path: &'static str, cnn_func: CNNSignature) -> Result<Self, TchError>
    {
        let device = Device::cuda_if_available();
        let mut varstore = VarStore::new(device);
        let model = cnn_func(&varstore.root(), imagenet::CLASS_COUNT);

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

    pub fn gen_embedding(&self, image_path: &PathBuf) -> Result<Tensor, TchError>
    {
        let embedding = imagenet::load_image(image_path)?;
        let embedding = embedding.to_device(self.varstore.device())
            .unsqueeze(0)
            .apply_t(&self.model, false)
            .softmax(-1, Kind::Float)
            .squeeze();

        Ok(embedding)
    }
}