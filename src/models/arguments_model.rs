use crate::errors::InvalidArgumentError;

pub struct ArgumentsModel
{
    target_dir: String
}

impl ArgumentsModel
{
    pub fn new(args: Vec<String>) -> Result<Self, InvalidArgumentError>
    {
        // Check if enough arguments are provided
        if args.len() < 2 {
            return Err(InvalidArgumentError("Not enough arguments provided"));
        }

        Ok(Self {
            target_dir: args[1].clone()
        })
    }

    pub fn target_dir(&self) -> &String
    {
        &self.target_dir
    }
}