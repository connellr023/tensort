use std::io;
use std::fs;
use std::path::PathBuf;
use tch::Tensor;
use crate::models::cnn_model::CNNModel;
use crate::Table;

pub fn extension_is_image(extension: &str) -> bool
{
    match extension.to_lowercase().as_str()
    {
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => true,
        _ => false
    }
}

pub fn gen_image_embeddings(dir: &PathBuf, model: &CNNModel) -> io::Result<(Vec<Tensor>, Vec<PathBuf>, Vec<PathBuf>)>
{
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