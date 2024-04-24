use std::error::Error;

#[derive(Debug)]
pub struct InvalidUsageError(pub &'static str);
impl Error for InvalidUsageError {}