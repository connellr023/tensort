use std::fmt::{Display, Formatter, Result};
use crate::errors::InvalidArgumentError;

impl Display for InvalidArgumentError
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result
    {
        write!(
            f,
            "Error: {}\n\
            \n\
            Usage: tensort <target_dir> <class_count> [-n | --gen-names]\n\
             \n\
             Arguments:\n\
             <target_dir>    : Path to the target directory\n\
             <class_count>   : Number of classes\n\
             -n, --gen-names : Generate class names (optional)\n\
             \n\
             Example:\n\
             tensort /path/to/images_dir 5 -n",
             self.0
        )
    }
}