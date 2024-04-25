use tch::vision::imagenet;
use tch::Tensor;

pub type Table<T> = Vec<Vec<T>>;

pub fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> f64
{
    let dot_product = t1.dot(t2);
    let norm1 = t1.norm();
    let norm2 = t2.norm();

    dot_product.double_value(&[]) / (norm1.double_value(&[]) * norm2.double_value(&[]))
}

pub fn calc_pairwise_cosine_similarities(embeddings: &[Tensor]) -> Vec<f64>
{
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

/// Formula for calculating the similarity threshold
/// for comparing image embeddings
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

/// The main sorting algorithm <br />
/// Given a slice of pre-computed pairwise similarities,
/// returns a table (2D vector) of embedding indicies that represent their "clusters"
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
                let similarity_index = first_embedding_index + (embedding_index * embedding_count);

                if similarity_index < similarities.len() && similarities[similarity_index] < threshold {
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

/// Takes a slice of tensors and returns a tensor which is an average of each dimension
pub fn calc_average_embedding(embeddings: &[&Tensor]) -> Tensor
{
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

/// Simply generates basic class names as a number representing order <br />
/// "Class 1, Class 2, ..., Class {n}"
pub fn gen_default_class_names(class_count: usize) -> Vec<String>
{
    let mut class_names = Vec::with_capacity(class_count);

    for i in 0..class_count {
        class_names.push(format!("Class {}", i + 1));
    }

    class_names
}

/// Given a slice of embeddings and the sorted table,
/// generate the most likely name for each classification (by averaging tensors)
pub fn gen_class_names(embeddings: &[Tensor], table: &Table<usize>) -> Vec<String>
{
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