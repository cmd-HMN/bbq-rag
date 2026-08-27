// common wrapper for test file

macro_rules! test_me {
    (
        $(#[$attr:meta])*
        $name: ident,
        $func: expr
        $(, $arg: expr)*
    ) => {
        $(#[$attr])*
        #[test]
        fn $name() {
            $func($($arg),*);
        }
    };

    // passign batch of test to it
    (
        group $grp: ident {
            $(
                $(#[$attr:meta])*
                $name: ident: $func: ident $(( $($arg:expr),* ))?
            ),* $(,)?
        }
    ) => {
        mod $grp {
            use super::*;
            $(
                $(#[$attr])*
                #[test]
                fn $name() {
                    $func($($arg),*);
                }
            )*
        }
    }
  
}
