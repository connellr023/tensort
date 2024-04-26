use std::io::Cursor;
use std::path::PathBuf;
use tch::{Device, Kind, TchError, Tensor};
use tch::nn::{Path, FuncT, VarStore};
use tch::vision::imagenet;

// Define a type alias for the CNN signature
pub type CNNSignature = fn(p: &Path, class_count: i64) -> FuncT<'static>;

// Define a trait for objects that can generate embeddings
pub trait Embeddable {

    /// Generates an embedding for the given image path.
    ///
    /// # Arguments
    ///
    /// * `image_path` - The path to the image file.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the generated embedding tensor, or an error if the embedding generation fails.
    fn gen_embedding(&self, image_path: &PathBuf) -> Result<Tensor, TchError>;
}

// Define the CNNModel struct
pub struct CNNModel {
    device: Device,
    varstore: VarStore,
    model: FuncT<'static>,
}

impl CNNModel {

    /// Creates a new CNNModel instance.
    ///
    /// # Arguments
    ///
    /// * `varstore_bytes` - The byte stream containing the serialized variable store.
    /// * `cnn_func` - The CNN function to use for model creation.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created CNNModel instance, or an error if the creation fails.
    pub fn new(varstore_bytes: &'static [u8], cnn_func: CNNSignature) -> Result<Self, TchError> {
        let device = Device::cuda_if_available();
        let mut varstore = VarStore::new(device);
        let model = cnn_func(&varstore.root(), imagenet::CLASS_COUNT);

        varstore.load_from_stream(Cursor::new(varstore_bytes))?;

        Ok(Self {
            device,
            varstore,
            model,
        })
    }

    /// Returns the device used by the CNNModel.
    pub fn device(&self) -> Device {
        self.device
    }
}

impl Embeddable for CNNModel {
    fn gen_embedding(&self, image_path: &PathBuf) -> Result<Tensor, TchError> {
        let embedding = imagenet::load_image_and_resize224(image_path)?;
        let embedding = embedding
            .to_device(self.varstore.device())
            .unsqueeze(0)
            .apply_t(&self.model, false)
            .softmax(-1, Kind::Float)
            .squeeze();

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use tch::Device;
    use tch::vision::resnet;
    use super::*;

    #[test]
    fn invalid_byte_stream_returns_error() {
        let invalid_byte_stream: &[u8] = &[69, 23];
        let result = CNNModel::new(invalid_byte_stream, resnet::resnet34);

        assert!(result.is_err());
    }

    #[test]
    fn valid_byte_stream_constructs_and_getters_work() {
        let valid_byte_stream: &[u8] = include_bytes!("../../resnet34.ot");
        let result = CNNModel::new(valid_byte_stream, resnet::resnet34).unwrap();

        assert_eq!(result.device(), Device::cuda_if_available());
    }
}