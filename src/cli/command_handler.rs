
use std::env::args;
use crate::cli::options::Options;

pub struct CommandHandler
{
    options: Options
}

impl CommandHandler
{
    pub fn new(args: Vec<String>) -> Self
    {
        Self {
            options: Options::from(args)
        }
    }
}