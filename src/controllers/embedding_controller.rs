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

pub fn calc_pairwise_cosine_similarities(embeddings: &[&Tensor]) -> Vec<f64>
{
    let embedding_count = embeddings.len();
    let mut similarities = Vec::with_capacity(embedding_count * embedding_count);

    for i in 0..embedding_count {
        for j in 0..embedding_count {
            let similarity = cosine_similarity(embeddings[i], embeddings[j]);
            similarities.push(similarity);
        }
    }

    similarities
}

pub fn calc_similarity_threshold(similarities: &[f64], class_count: usize) -> f64
{
    let similarities_per_class = similarities.len() / class_count;
    let mut max_within_clusters = Vec::with_capacity(class_count);

    // Calculate maxmimum similarity within each cluster
    for i in 0..class_count {
        let start = i * similarities_per_class;
        let end = start + similarities_per_class;
        let max = similarities[start..end]
            .iter()
            .cloned()
            .fold(
                std::f64::NEG_INFINITY,
                |max, similarity| { max.max(similarity) }
        );

        max_within_clusters.push(max);
    }

    // Calculate minimum similarity between clusters
    let min_between_clusters = max_within_clusters
        .iter()
        .fold(
            std::f64::INFINITY,
            |min, &max_similarity| { min.min(max_similarity) }
    );

    let sum = max_within_clusters
        .iter()
        .sum::<f64>();

    ((sum / class_count as f64) + min_between_clusters) / 2.0
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