use std::fmt::{Display, Formatter, Result};
use std::path::PathBuf;
use crate::controllers::embeddings_controller::Table;

struct MissedImagesFormatter(Vec<PathBuf>);

impl Display for MissedImagesFormatter {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        writeln!(f, "\nThe following images failed to process:")?;

        for path in &self.0 {
            writeln!(f, "\t=> {}", path
                .to_str()
                .unwrap()
            )?;
        }

        Ok(())
    }
}
struct ClassifiedImagesFormatter {
    similarity_table: Table<usize>,
    image_paths: Vec<PathBuf>,
    class_names: Vec<String>
}

impl ClassifiedImagesFormatter {
    fn new(
        similarity_table: Table<usize>,
        image_paths: Vec<PathBuf>,
        class_names: Vec<String>
    ) -> Self
    {
        Self {
            similarity_table,
            image_paths,
            class_names
        }
    }
}

impl Display for ClassifiedImagesFormatter
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for i in 0..self.similarity_table.len() {
            writeln!(f, "{}:", self.class_names[i])?;
    
            for j in 0..self.similarity_table[i].len() {
                writeln!(f, "\t=> {}", self.image_paths[self.similarity_table[i][j]]
                    .to_str()
                    .unwrap()
                )?;
            }
    
            writeln!(f)?;
        }

        Ok(())
    }
}

pub fn format_missed_images(missed_images: Vec<PathBuf>) -> impl Display
{
    MissedImagesFormatter(missed_images)
}

pub fn format_classified_images(
    similarity_table: Table<usize>,
    image_paths: Vec<PathBuf>,
    class_names: Vec<String>
) -> impl Display
{
    ClassifiedImagesFormatter::new(similarity_table, image_paths, class_names)
}