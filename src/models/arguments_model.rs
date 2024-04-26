use std::path::PathBuf;
use crate::errors::InvalidUsageError;

/// Minimum number of arguments that should be provided to the CLI <br />
/// Required: <br />
///     <target_dir> <br />
///     <class_count> <br />
/// Optional: <br />
///     <no_names> (default = false)
const MIN_ARG_COUNT: usize = 3;

#[derive(PartialEq, Debug)]
pub struct ArgumentsModel
{
    target_dir: PathBuf,
    class_count: usize,
    should_not_gen_names: bool
}

impl ArgumentsModel
{
    /// Factory method for a new arguments structure <br />
    /// Represents the values passed to the CLI in the context of this tool
    pub fn from(args: Vec<String>) -> Result<Self, InvalidUsageError>
    {
        // Check if enough arguments are provided
        if args.len() < MIN_ARG_COUNT {
            return Err(InvalidUsageError("Not enough arguments provided"));
        }

        let target_dir = PathBuf::from(args[1].clone());
        if !target_dir.is_dir() {
            return Err(InvalidUsageError("Provided path is not a directory"));
        }

        let class_count = match args[2].parse::<usize>() {
            Ok(class_count) => class_count,
            Err(_) => {
                return Err(InvalidUsageError("Invalid number provided for class count"));
            }
        };

        let gen_names = match args.get(3) {
            Some(flag) => {
                match flag.as_str() {
                    "-n" | "--no-names" => true,
                    _ => false
                }
            },
            None => false
        };

        Ok(Self {
            target_dir,
            class_count,
            should_not_gen_names: gen_names,
        })
    }

    pub fn target_dir(&self) -> &PathBuf
    {
        &self.target_dir
    }

    pub fn class_count(&self) -> usize
    {
        self.class_count
    }

    pub fn should_not_gen_names(&self) -> bool
    {
        self.should_not_gen_names
    }
}

#[cfg(test)]
mod tests
{
    use assertables::*;
    use crate::models::arguments_model::ArgumentsModel;

    #[test]
    fn not_enough_args_returns_error()
    {
        let result = ArgumentsModel::from(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_dir_returns_error()
    {
        let not_a_dir = std::env::current_dir()
            .unwrap()
            .join("mod.rs")
            .to_str()
            .unwrap()
            .to_string();

        let result = ArgumentsModel::from(vec![String::from("tensort"), not_a_dir, String::from("5")]);

        match result {
            Ok(_) => assert!(false),
            Err(err) => {
                let message = err.to_string();
                println!("{}", message);
                assert_contains!(message, "not a directory");
            }
        }
    }

    #[test]
    fn invalid_class_count_returns_error()
    {
        let valid_dir = std::env::current_dir()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let result = ArgumentsModel::from(vec![String::from("tensort"), valid_dir, String::from("-3")]);

        assert!(result.is_err());
    }

    #[test]
    fn valid_input_no_class_names_constructs()
    {
        let valid_dir_path = std::env::current_dir().unwrap();
        let valid_dir = valid_dir_path
            .to_str()
            .unwrap()
            .to_string();

        let result = ArgumentsModel::from(vec![String::from("tensort"), valid_dir, String::from("4"), String::from("-n")]).unwrap();
        let expected = ArgumentsModel {
            class_count: 4,
            target_dir: valid_dir_path,
            should_not_gen_names: true
        };

        assert_eq!(expected, result);
    }

    #[test]
    fn valid_input_with_class_names_constructs()
    {
        let valid_dir_path = std::env::current_dir().unwrap();
        let valid_dir = valid_dir_path
            .to_str()
            .unwrap()
            .to_string();

        let result = ArgumentsModel::from(vec![String::from("tensort"), valid_dir, String::from("8")]).unwrap();
        let expected = ArgumentsModel {
            class_count: 8,
            target_dir: valid_dir_path,
            should_not_gen_names: false
        };

        assert_eq!(expected, result);
    }

    #[test]
    fn getters_work()
    {
        let valid_dir_path = std::env::current_dir().unwrap();
        let valid_dir = valid_dir_path
            .to_str()
            .unwrap()
            .to_string();

        let result = ArgumentsModel::from(vec![String::from("tensort"), valid_dir, String::from("8")]).unwrap();

        assert_eq!(*result.target_dir(), valid_dir_path);
        assert_eq!(result.class_count(), 8);
        assert_eq!(result.should_not_gen_names(), false);
    }
}