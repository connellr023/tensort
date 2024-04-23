use std::path::PathBuf;
use std::io::{Error, ErrorKind, Result};
use std::fs::read_dir;
use tch::Tensor;
use crate::models::cnn_model::CNNModel;

pub type TensorPathTuple = (Tensor, PathBuf);
pub type Table<T> = Vec<Vec<T>>;

pub fn extension_is_image(extension: &str) -> bool
{
    match extension.to_lowercase().as_str()
    {
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => true,
        _ => false
    }
}

pub fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> f64
{
    let dot_product = t1.dot(t2);
    let norm1 = t1.norm();
    let norm2 = t2.norm();

    dot_product.double_value(&[]) / (norm1.double_value(&[]) * norm2.double_value(&[]))
}

pub fn calc_pairwise_cosine_similarities(embeddings: &[TensorPathTuple]) -> Vec<f64>
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

    (((sum / class_count as f64) + min_between_clusters)) / 2.0
}

pub fn gen_image_embeddings(dir: &PathBuf, model: &CNNModel) -> Result<(Vec<TensorPathTuple>, Vec<PathBuf>)>
{
    let mut embeddings = vec![];
    let mut missed_images = vec![];

    if !dir.is_dir() {
        return Err(Error::new(ErrorKind::InvalidInput, "Path supplied is not a directory"));
    }

    for file in read_dir(dir)? {
        let file = file?;
        let path = file.path();

        if let Some(extension) = path.extension().and_then(|extension| { extension.to_str() }) {
            if extension_is_image(extension) {
                match model.gen_embedding(&path) {
                    Ok(embedding) => embeddings.push((embedding, path)),
                    Err(_) => missed_images.push(path)
                }
            }
        }
    }

    Ok((embeddings, missed_images))
}

pub fn cluster_embeddings(similarities: &[f64], embedding_count: usize, class_count: usize) -> Table<usize>
{
    let threshold = calc_similarity_threshold(similarities, class_count);
    let mut clusters: Table<usize> = vec![vec![]; class_count];
    let mut last_class_index = 0usize;
    let mut last_embedding_index = 0usize;

    // Main loop for assigning initial embedding indicies to each class in the table
    for embedding_index in 0..embedding_count {
        last_embedding_index = embedding_index;

        // Check if all available classes have at least one embedding assigned to them
        // Stop this main assignment loop otherwise
        if last_class_index >= class_count {
            break;
        }

        for class_index in 0..class_count {
            if let Some(first_embedding_index) = clusters[class_index].first() {

                // Calculate row major order index offset
                let similarity_index_offset = first_embedding_index + (embedding_index * embedding_count);

                if similarity_index_offset < similarities.len() && similarities[similarity_index_offset] < threshold {
                    continue;
                }
            }

            clusters[class_index].push(embedding_index);
            last_class_index += 1;

            break;
        }
    }

    // Second loop for assigning overflowed embedding indicies as they best fit
    for embedding_index in last_embedding_index..embedding_count {

        // Track best as a tuple of its current class index and current best cosine similarity
        // Initialize cosine similarity to -1.0 which represents completely opposing vectors
        let mut best = (0, -1.0);
        
        // Probe the table to find which classification the overflowed embeddings fit best into
        for class_index in 0..class_count {
            if let Some(first_embedding_index) = clusters[class_index].first() {
                let similarity_index_offset = first_embedding_index + (embedding_index * embedding_count);
                let similarity = similarities[similarity_index_offset];

                // Update best if the current is better than previous best
                if similarity > best.1 {
                    best = (class_index, similarity);
                }
            }
        }

        clusters[best.0].push(embedding_index);
    }

    clusters
}