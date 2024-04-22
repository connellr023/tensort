use std::path::PathBuf;
use std::io::Result;
use std::fs::read_dir;
use tch::Tensor;
use crate::models::resnet18_model::ResNet18Model;

pub type TensorPathTuple = (Tensor, PathBuf);

fn extension_is_image(extension: &str) -> bool
{
    match extension.to_lowercase().as_str()
    {
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => true,
        _ => false
    }
}

fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> f64
{
    let dot_product = t1.dot(t2);
    let norm1 = t1.norm();
    let norm2 = t2.norm();

    dot_product.double_value(&[]) / (norm1.double_value(&[]) * norm2.double_value(&[]))
}

pub fn calc_pairwise_cosine_similarities(embeddings: &Vec<TensorPathTuple>) -> Vec<f64>
{
    let embedding_count = embeddings.len();
    let mut similarities = Vec::with_capacity(embedding_count * embedding_count);

    for i in 0..embedding_count {
        for j in 0..embedding_count {
            let similarity = cosine_similarity(&embeddings[i].0, &embeddings[j].0);
            similarities.push(similarity);
        }
    }

    similarities
}

pub fn gen_image_embeddings(dir: &PathBuf, model: &ResNet18Model) -> Result<Vec<TensorPathTuple>>
{
    let mut embeddings = vec![];

    if !dir.is_dir() {
        println!("Not dir");
    }

    for file in read_dir(dir)? {
        let file = file?;
        let path = file.path();

        if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
            if extension_is_image(extension) {
                if let Ok(embedding) = model.gen_embedding(&path) {
                    embeddings.push((embedding, path));
                }
            }
        }
    }

    Ok(embeddings)
}