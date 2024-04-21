use std::error::Error;
use std::fmt::{
    Display,
    Formatter,
    Result
};

macro_rules! def_error
{
    ($error_name:ident) => {
        #[derive(Debug)]
        pub struct $error_name(pub &'static str);

        impl Display for $error_name
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result
            {
                write!(f, "{}: {}", stringify!($error_name), self.0)
            }
        }

        impl Error for $error_name {}
    };
}

def_error!(InvalidArgumentError);