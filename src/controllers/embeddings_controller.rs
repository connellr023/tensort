use tch::vision::imagenet;
use tch::Tensor;

pub type Table<T> = Vec<Vec<T>>;

/// Calculates the cosine similarity between two tensors.
pub fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> f64 {
    let dot_product = t1.dot(t2);
    let norm1 = t1.norm();
    let norm2 = t2.norm();

    dot_product.double_value(&[]) / (norm1.double_value(&[]) * norm2.double_value(&[]))
}

/// Calculates the pairwise cosine similarities between a slice of embeddings.
pub fn calc_pairwise_cosine_similarities(embeddings: &[Tensor]) -> Vec<f64> {
    let embedding_count = embeddings.len();
    let mut similarities = Vec::with_capacity(embedding_count * embedding_count);

    for i in 0..embedding_count {
        for j in 0..embedding_count {
            let similarity = cosine_similarity(&embeddings[i], &embeddings[j]);
            similarities.push(similarity);
        }
    }

    similarities
}

/// Calculates the similarity threshold for comparing image embeddings.
pub fn calc_similarity_threshold(similarities: &[f64], class_count: usize) -> f64 {
    let similarities_per_class = similarities.len() / class_count;
    let mut max_within_clusters = Vec::with_capacity(class_count);

    // Calculate maximum similarity within each cluster
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

/// Clusters embeddings based on their similarities.
///
/// This function takes a slice of similarities between embeddings, the total number of embeddings, and the desired number of classes (clusters). It uses these inputs to perform a clustering operation, grouping similar embeddings together.
///
/// # Arguments
///
/// * `similarities` - A slice of `f64` values representing the similarities between different embeddings. The length of this slice should be equal to the square of `embedding_count`.
/// * `similarity_threshold` - The threshold for considering two embeddings as similar. This value is calculated based on the similarities between embeddings and the desired number of classes.
/// * `embedding_count` - The total number of embeddings that are being clustered.
/// * `class_count` - The desired number of clusters.
///
/// # Returns
///
/// This function returns a `Table<usize>` where each row represents an embedding and each column represents a class. The value at a specific row and column indicates the membership of the corresponding embedding in the corresponding class.
pub fn cluster_embeddings(similarities: &[f64], similarity_threshold: f64, embedding_count: usize, class_count: usize) -> Table<usize> {
    let mut clusters: Table<usize> = vec![vec![]; class_count];
    let mut last_class_index = 0usize;
    let mut last_embedding_index = 0usize;

    // Main loop for assigning initial embedding indices to each class in the table
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
                let similarity_index = first_embedding_index + (embedding_index * embedding_count);

                if similarity_index < similarities.len() && similarities[similarity_index] < similarity_threshold {
                    continue;
                }
            }

            clusters[class_index].push(embedding_index);
            last_class_index += 1;

            break;
        }
    }

    // Second loop for assigning overflowed embedding indices as they best fit
    for embedding_index in last_embedding_index..embedding_count {

        // Track best as a tuple of its current class index and current best cosine similarity
        // Initialize cosine similarity to -1.0 which represents completely opposing vectors
        let mut best = (0, -1.0);
        
        // Probe the table to find which classification the overflowed embeddings fit best into
        for class_index in 0..class_count {
            if let Some(first_embedding_index) = clusters[class_index].first() {
                let similarity_index = first_embedding_index + (embedding_index * embedding_count);
                let similarity = similarities[similarity_index];

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

/// Takes a slice of tensors and returns a tensor which is an average of each dimension.
pub fn calc_average_embedding(embeddings: &[&Tensor]) -> Tensor {
    let mut tensor_sum = Tensor::zeros_like(&embeddings[0]);

    embeddings
        .iter()
        .for_each(|embedding| {

            // Perform vector addition on each tensor
            tensor_sum += *embedding;
        });

    // Calculate average of each dimension
    tensor_sum / embeddings.len() as f64
}

/// Generates basic class names as a number representing order.
/// Example: "Class 1, Class 2, ..., Class {n}".
pub fn gen_default_class_names(class_count: usize) -> Vec<String> {
    let mut class_names = Vec::with_capacity(class_count);

    for i in 0..class_count {
        class_names.push(format!("Class {}", i + 1));
    }

    class_names
}

/// Given a slice of embeddings and the sorted table,
/// generates the most likely name for each classification (by averaging tensors).
pub fn gen_class_names(embeddings: &[Tensor], table: &Table<usize>) -> Vec<String> {
    let class_count = table.len();
    let mut class_names = Vec::with_capacity(class_count);

    for i in 0..class_count {
        let row = &table[i];

        // Simply fill the rest of the vector since the entire space needs to be used
        if row.len() == 0 {
            class_names.push(format!("Empty class ({})", i + 1));
            continue;
        }

        let row_embeddings: Vec<&Tensor> = row
            .iter()
            .map(|embedding_index| { &embeddings[*embedding_index] })
            .collect();

        let average_embedding = calc_average_embedding(row_embeddings.as_slice());
        let class_name = match imagenet::top(&average_embedding, 1).first() {
            Some(top_class_name) => format!("{} ({})", top_class_name.1, i + 1),
            None => String::new()
        };

        class_names.push(class_name);
    }

    class_names
}

#[cfg(test)]
mod tests {
    use assertables::*;
    use super::*;

    #[test]
    fn cosine_similarity_with_same_tensors_works() {
        let t1 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let t2 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let similarity = cosine_similarity(&t1, &t2);

        assert_in_delta!(similarity, 1.0, 1e-6);
    }

    #[test]
    fn cosine_similarity_with_opposing_tensors_works() {
        let t1 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let t2 = Tensor::from_slice(&[-4.0, -5.0, -6.0]);
        let similarity = cosine_similarity(&t1, &t2);

        assert_in_delta!(similarity, -1.0, 1e-6);
    }

    #[test]
    fn cosine_similarity_with_orthogonal_tensors_works() {
        let t1 = Tensor::from_slice(&[1.0, 0.0, 0.0]);
        let t2 = Tensor::from_slice(&[0.0, 1.0, 0.0]);
        let similarity = cosine_similarity(&t1, &t2);

        assert_in_delta!(similarity, 0.0, 1e-6);
    }

    #[test]
    fn calc_pairwise_cosine_similarities_works() {
        let t1 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let t2 = Tensor::from_slice(&[-4.0, -5.0, -6.0]);
        let embeddings = vec![t1, t2];
        let similarities = calc_pairwise_cosine_similarities(embeddings.as_slice());

        assert_eq!(similarities.len(), 4);

        assert_in_delta!(similarities[0], 1.0, 1e-6);
        assert_in_delta!(similarities[1], -1.0, 1e-6);
        assert_in_delta!(similarities[2], -1.0, 1e-6);
        assert_in_delta!(similarities[3], 1.0, 1e-6);
    }

    #[test]
    fn calc_similarity_threshold_within_range() {
        let similarities = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let class_count = 3;
        let threshold = calc_similarity_threshold(similarities.as_slice(), class_count);
    
        assert_le!(threshold, 1.0);
        assert_ge!(threshold, -1.0);
    }

    #[test]
    fn cluster_embeddings_works() {
        let similarities = vec![
            0.9, 0.8, 0.7,
            0.8, 0.9, 0.8,
            0.7, 0.8, 0.9
        ];

        let similarity_threshold = 0.9;
        let embedding_count = 3;
        let class_count = 2;
        let table = cluster_embeddings(similarities.as_slice(), similarity_threshold, embedding_count, class_count);

        assert_eq!(table.len(), class_count);
        assert_eq!(table[0], vec![0]);
        assert_eq!(table[1], vec![1, 2]);
    }

    #[test]
    fn gen_default_class_names_works() {
        let class_count = 5;
        let class_names = gen_default_class_names(class_count);

        assert_eq!(class_names.len(), class_count);
        assert_eq!(class_names[0], "Class 1");
        assert_eq!(class_names[1], "Class 2");
        assert_eq!(class_names[2], "Class 3");
        assert_eq!(class_names[3], "Class 4");
        assert_eq!(class_names[4], "Class 5");
    }

    #[test]
    fn gen_class_names_works() {
        let mut slice1 = [0.0; 1000];
        let mut slice2 = [0.0; 1000];

        slice1[0] = 1.0;
        slice2[1] = 1.0;

        let t1 = Tensor::from_slice(&slice1);
        let t2 = Tensor::from_slice(&slice2);
        let embeddings = vec![t1, t2];
        let table = vec![vec![0], vec![1]];
        let class_names = gen_class_names(embeddings.as_slice(), &table);

        assert_eq!(class_names.len(), 2);

        assert_contains!(class_names[0], "tench");
        assert_contains!(class_names[1], "goldfish");
    }
}