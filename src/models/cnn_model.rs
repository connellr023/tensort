use std::path::PathBuf;
use tch::{Device, Kind, TchError, Tensor};
use tch::nn::{Path, FuncT, VarStore};
use tch::vision::imagenet;
use std::io::Cursor;

pub type CNNSignature = fn(p: &Path, class_count: i64) -> FuncT<'static>;

pub struct CNNModel
{
    device: Device,
    varstore: VarStore,
    model: FuncT<'static>
}

impl CNNModel
{
    pub fn new(model_weight_bytes: &'static [u8], cnn_func: CNNSignature) -> Result<Self, TchError>
    {
        let device = Device::cuda_if_available();
        let mut varstore = VarStore::new(device);
        let model = cnn_func(&varstore.root(), imagenet::CLASS_COUNT);

        varstore.load_from_stream(Cursor::new(model_weight_bytes))?;
        //varstore.load(pretrained_path)?;
        
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
        let embedding = imagenet::load_image_and_resize224(image_path)?;
        let embedding = embedding.to_device(self.varstore.device())
            .unsqueeze(0)
            .apply_t(&self.model, false)
            .softmax(-1, Kind::Float)
            .squeeze();

        Ok(embedding)
    }
}