use std::fmt::{Display, Formatter, Result};
use crate::errors::InvalidUsageError;

impl Display for InvalidUsageError
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result
    {
        write!(
            f,
            "Error: {}\n\
            \n\
            Usage: tensort <target_dir> <class_count> [-n | --no-names]\n\
             \n\
             Arguments:\n\
             <target_dir>    : Path to the target directory\n\
             <class_count>   : Number of classes\n\
             -n, --no-names  : Do not generate class names (optional)\n\
             \n\
             Example:\n\
             tensort /path/to/images_dir 5 -n",
             self.0
        )
    }
}