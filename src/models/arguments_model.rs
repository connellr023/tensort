use std::path::PathBuf;
use crate::errors::InvalidArgumentError;

pub struct ArgumentsModel
{
    target_dir: PathBuf
}

impl ArgumentsModel
{
    pub fn new(args: Vec<String>) -> Result<Self, InvalidArgumentError>
    {
        // Check if enough arguments are provided
        if args.len() < 2 {
            return Err(InvalidArgumentError("Not enough arguments provided"));
        }

        let target_dir = PathBuf::from(args[1].clone());
        if !target_dir.is_dir() {
            return Err(InvalidArgumentError("Provided path is not a directory"));
        }

        Ok(Self {
            target_dir
        })
    }

    pub fn target_dir(&self) -> &PathBuf
    {
        &self.target_dir
    }
}