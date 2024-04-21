use std::error::Error;
use std::fmt::{
    Display,
    Formatter,
    Result
};

#[derive(Debug)]
pub struct HandlerError(pub &'static str);

impl Display for HandlerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result
    {
        write!(f, "{}", self.0)
    }
}

impl Error for HandlerError {}