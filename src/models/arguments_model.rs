use std::path::PathBuf;
use crate::errors::InvalidUsageError;

/// Minimum number of arguments that should be provided to the CLI <br />
/// Required: <br />
///     <target_dir> <br />
///     <class_count> <br />
/// Optional: <br />
///     <should_gen_names> (default = false)
const MIN_ARG_COUNT: usize = 3;

pub struct ArgumentsModel
{
    target_dir: PathBuf,
    class_count: usize,
    should_gen_names: bool
}

impl ArgumentsModel
{
    /// Factory method for a new arguments structure <br />
    /// Represents the values passed to the CLI in the context of this tool
    pub fn new(args: Vec<String>) -> Result<Self, InvalidUsageError>
    {
        // Check if enough arguments are provided
        if args.len() < MIN_ARG_COUNT {
            return Err(InvalidUsageError("Not enough arguments provided"));
        }

        let target_dir = PathBuf::from(args[1].clone());
        if !target_dir.is_dir() {
            return Err(InvalidUsageError("Provided path is not a directory"));
        }

        let class_count = match args[2].parse::<usize>() {
            Ok(class_count) => class_count,
            Err(_) => {
                return Err(InvalidUsageError("Invalid number provided for class count"));
            }
        };

        let gen_names = match args.get(3) {
            Some(flag) => {
                match flag.as_str() {
                    "-n" | "--gen-names" => true,
                    _ => false
                }
            },
            None => false
        };

        Ok(Self {
            target_dir,
            class_count,
            should_gen_names: gen_names,
        })
    }

    pub fn target_dir(&self) -> &PathBuf
    {
        &self.target_dir
    }

    pub fn class_count(&self) -> usize
    {
        self.class_count
    }

    pub fn should_gen_names(&self) -> bool
    {
        self.should_gen_names
    }
}