use std::fmt::{Display, Formatter, Result};
use crate::models::arguments_model::ArgumentsModel;

impl Display for ArgumentsModel
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Running tensort with options:\n\
            <target_dir>        : {}\n\
            <class_count>       : {}\n\
            <no_class_names>    : {}
            ",
            self.target_dir().to_str().unwrap(),
            self.class_count(),
            self.should_not_gen_names()
        )
    }
}