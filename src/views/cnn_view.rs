use std::fmt::{Display, Formatter, Result};
use crate::models::cnn_model::CNNModel;

impl Display for CNNModel
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result
    {
        write!(
            f,
            "Neural network running on device: {:?}",
            self.device()
        )
    }
}