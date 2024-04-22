use std::error::Error;

#[derive(Debug)]
pub struct InvalidArgumentError(pub &'static str);
impl Error for InvalidArgumentError {}