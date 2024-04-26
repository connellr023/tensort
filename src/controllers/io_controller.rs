use std::io;
use std::fs;
use std::path::PathBuf;
use tch::Tensor;
use crate::models::cnn_model::Embeddable;
use crate::Table;

/// Checks if the given file extension is an image extension.
///
/// # Arguments
///
/// * `extension` - A string slice representing the file extension.
///
/// # Returns
///
/// Returns `true` if the extension is a valid image extension (jpg, jpeg, or png), otherwise `false`.
fn extension_is_image(extension: &str) -> bool {
    match extension.to_lowercase().as_str()
    {
        "jpg" | "jpeg" | "png" => true,
        _ => false
    }
}

/// Generates image embeddings for each image in a directory.
///
/// This function takes a directory and a model that implements the `Embeddable` trait.
/// It iterates over each file in the directory. If the file is an image (determined by the `extension_is_image` function),
/// it generates an embedding using the `gen_embedding` method of the model.
///
/// # Arguments
///
/// * `dir` - A `PathBuf` that represents the directory to search for images.
/// * `model` - A reference to an instance of a model that implements the `Embeddable` trait.
///
/// # Returns
///
/// This function returns a `Result` with a tuple containing three vectors:
///
/// * A vector of `Tensor` objects, each representing the embedding of an image.
/// * A vector of `PathBuf` objects, each representing the path of an image that was successfully processed.
/// * A vector of `PathBuf` objects, each representing the path of an image that could not be processed.
///
/// If the `dir` argument is not a directory, the function returns an `Err` with an `io::Error` of kind `InvalidInput`.
pub fn gen_image_embeddings<T: Embeddable>(dir: &PathBuf, model: &T) -> io::Result<(Vec<Tensor>, Vec<PathBuf>, Vec<PathBuf>)> {
    let mut embeddings = vec![];
    let mut images_paths = vec![];
    let mut missed_images_paths = vec![];

    if !dir.is_dir() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Path supplied is not a directory"));
    }

    for file in fs::read_dir(dir)? {
        let file = file?;
        let path = file.path();

        if let Some(extension) = path.extension().and_then(|extension| { extension.to_str() }) {
            if extension_is_image(extension) {
                match model.gen_embedding(&path) {
                    Ok(embedding) => {
                        embeddings.push(embedding);
                        images_paths.push(path);
                    },
                    Err(_) => missed_images_paths.push(path)
                }
            }
        }
    }

    Ok((embeddings, images_paths, missed_images_paths))
}

/// Uses the table generated from sorting and the generated class names.
/// Will re-arrange the target `dir` to reflect the sorting results.
///
/// # Arguments
///
/// * `dir` - A `PathBuf` that represents the target directory to be updated.
/// * `image_paths` - A slice of `PathBuf` objects representing the paths of the images to be moved.
/// * `class_names` - A slice of `String` objects representing the class names.
/// * `table` - A reference to a `Table<usize>` object representing the sorting results.
///
/// # Returns
///
/// Returns `Ok(())` if the target directory is successfully updated, otherwise returns an `io::Error`.
pub fn update_target_dir(
    dir: &PathBuf,
    image_paths: &[PathBuf],
    class_names: &[String],
    table: &Table<usize>
) -> io::Result<()>
{
    let class_count = table.len();

    for i in 0..class_count {
        let row = &table[i];

        // Do not continue if there are empty classes
        if row.len() == 0 {
            break;
        }

        let class_dir = &dir.join(&class_names[i]);

        // Create the directory with the desired class name
        fs::create_dir(class_dir)?;

        // Move images into the new directory
        for &image_path_index in row {
            let src_image_path = &image_paths[image_path_index];
            let dest_image_path = &class_dir.join(src_image_path.file_name().unwrap());

            // Move the file
            fs::rename(src_image_path, dest_image_path)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    struct MockCNNModel;

    impl Embeddable for MockCNNModel {
        fn gen_embedding(&self, _: &PathBuf) -> Result<Tensor, tch::TchError> {
            Ok(Tensor::new())
        }
    }

    #[test]
    fn extension_is_image_works() {
        assert!(extension_is_image("png"));
        assert!(extension_is_image("PNG"));
        assert!(extension_is_image("jpg"));
        assert!(extension_is_image("JPG"));
        assert!(extension_is_image("jpeg"));
        assert!(extension_is_image("JPEG"));
    }

    #[test]
    fn extension_is_not_image_works() {
        assert!(!extension_is_image("txt"));
        assert!(!extension_is_image("BMP"));
    }

    #[test]
    fn gen_image_embeddings_with_valid_input_works() {
        let model = MockCNNModel;
        let dir = tempdir().unwrap();
        let img_path = dir.path().join("image.jpg");
        let non_img_path = dir.path().join("file.txt");

        // Create an image file and a non-image file
        File::create(&img_path).unwrap();
        let mut file = File::create(&non_img_path).unwrap();
        writeln!(file, "Hello, world!").unwrap();

        let (embeddings, images_paths, missed_images_paths) = gen_image_embeddings(&dir.path().to_path_buf(), &model).unwrap();

        // Check that the image file was processed and the non-image file was not
        assert_eq!(embeddings.len(), 1);
        assert_eq!(images_paths, vec![img_path]);
        assert_eq!(missed_images_paths.len(), 0);
    }

    #[test]
    fn gen_image_embeddings_with_invalid_input_works() {
        let model = MockCNNModel;
        let dir = tempdir().unwrap();
        let img_path = dir.path().join("image.jpg");

        // Create an image file
        File::create(&img_path).unwrap();

        // Attempt to process a non-existent directory
        let result = gen_image_embeddings(&dir.path().join("non_existent"), &model);
        assert!(result.is_err());

        // Attempt to process a non-directory
        let result = gen_image_embeddings(&img_path, &model);
        assert!(result.is_err());
    }
}