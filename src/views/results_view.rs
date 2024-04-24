use std::fmt::{Display, Formatter, Result};
use std::path::PathBuf;

use crate::controllers::embeddings_controller::{Table, TensorPathTuple};

struct MissedImagesFormatter(Vec<PathBuf>);

impl Display for MissedImagesFormatter
{
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

struct ClassifiedImagesFormatter
{
    similarity_table: Table<usize>,
    embeddings: Vec<TensorPathTuple>,
    class_names: Vec<Option<String>>
}

impl ClassifiedImagesFormatter
{
    fn new(
        similarity_table: Table<usize>,
        embeddings: Vec<TensorPathTuple>,
        class_names: Vec<Option<String>>
    ) -> Self
    {
        Self {
            similarity_table,
            embeddings,
            class_names
        }
    }
}

impl Display for ClassifiedImagesFormatter
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for i in 0..self.similarity_table.len() {
            let class_name = match self.class_names.get(i) {
                Some(class_name_opt) => match class_name_opt {
                    Some(class_name) => Some(class_name),
                    None => None
                },
                None => None
            };

            match class_name {
                Some(class_name) => {
                    writeln!(f, "Class {} - {}:", i, class_name)?;
                },
                None => {
                    writeln!(f, "Class {}:", i)?;
                }
            }
    
            for j in 0..self.similarity_table[i].len() {
                writeln!(f, "\t=> {}", self.embeddings[self.similarity_table[i][j]]
                    .1
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
    embeddings: Vec<TensorPathTuple>,
    class_names: Vec<Option<String>>
) -> impl Display
{
    ClassifiedImagesFormatter::new(similarity_table, embeddings, class_names)
}